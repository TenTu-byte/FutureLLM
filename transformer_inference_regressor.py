# -*- coding: utf-8 -*-
"""
transformer_inference_regressor.py (FAST)

✅ Same external behavior as your original script:
- Same CLI args / defaults
- Same dataset reading
- Same sharding, output path, output jsonl fields, checkpoint fields, insert policy text, etc.

✅ Only optimize inference logic:
1) Reuse past_key_values every step (incremental decoding; no generate()).
2) When token hits "\n\n" checkpoint (token contains "ĊĊ"), directly take CURRENT step hidden
   at reg_layer and feed regressor (no extra forward over full sequence).
3) Once insertion triggers, immediately generate the rest to EOS / budget (no further checkpoint checks).

Notes:
- We intentionally keep output schema identical to your original version.
- Sampling behavior matches HF generate() for the common case (temperature/top_p/top_k/min_p).
"""

import os
import json
import argparse
import gc
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import joblib

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ------------------------
# Small utils
# ------------------------

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_indices(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "idx" in obj:
                    done.add(int(obj["idx"]))
            except Exception:
                continue
    return done


# ------------------------
# Stop-id builder (tokens containing "ĊĊ" for "\n\n")
# ------------------------

def build_stop_ids_contains(tok, needle: str = "ĊĊ"):
    vocab = tok.get_vocab()  # token_str -> id
    stop_ids = {int(i) for s, i in vocab.items() if needle in s}
    if not stop_ids:
        raise RuntimeError(f'No tokens containing "{needle}" found in tokenizer vocab.')
    return stop_ids


# ------------------------
# Model load
# ------------------------

def load_model_tokenizer(model_name_or_path, trust_remote_code, device):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok


# ------------------------
# Regressor utils
# ------------------------

def safe_predict_regressor(reg, x_1d: np.ndarray) -> float:
    X = x_1d.reshape(1, -1)
    y = reg.predict(X)
    y = float(np.asarray(y).reshape(-1)[0])
    return y


def maybe_clip_pred(pred: float, norm_mode: str) -> float:
    if norm_mode in ("log", "minmax"):
        if pred < 0.0:
            return 0.0
        if pred > 1.0:
            return 1.0
        return float(pred)
    return float(pred)


# ------------------------
# Sampling (generate()-like)
# ------------------------

def _top_k_filtering(logits: torch.Tensor, top_k: int):
    if top_k is None or int(top_k) <= 0:
        return logits
    top_k = int(top_k)
    if top_k >= logits.shape[-1]:
        return logits
    v, _ = torch.topk(logits, top_k, dim=-1)
    kth = v[..., -1, None]
    return torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)


def _top_p_filtering(logits: torch.Tensor, top_p: float):
    if top_p is None:
        return logits
    top_p = float(top_p)
    if top_p >= 1.0 or top_p <= 0.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    # keep minimal set with cumulative prob <= top_p
    cutoff = cumprobs > top_p
    # ensure at least 1 token kept
    cutoff[..., 0] = False
    sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits)

    # scatter back
    new_logits = torch.full_like(logits, -float("inf"))
    new_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return new_logits


def _min_p_filtering(logits: torch.Tensor, min_p: float):
    """
    Approximate HF MinPLogitsWarper behavior:
    keep tokens with prob >= min_p * max_prob (after temperature scaling).
    """
    if min_p is None:
        return logits
    min_p = float(min_p)
    if min_p <= 0.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    keep = probs >= (min_p * max_prob)
    # ensure at least 1 token kept
    keep[..., probs.argmax(dim=-1, keepdim=True)] = True
    return torch.where(keep, logits, torch.full_like(logits, -float("inf")))


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    """
    logits: [1, vocab]
    returns token_id: int
    """
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())

    temp = float(temperature) if temperature is not None else 1.0
    if temp <= 0:
        # degenerate -> greedy
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temp
    logits = _top_k_filtering(logits, top_k)
    logits = _top_p_filtering(logits, top_p)
    logits = _min_p_filtering(logits, min_p)

    probs = torch.softmax(logits, dim=-1)
    # numerical safety
    if torch.isnan(probs).any() or torch.isinf(probs).any() or float(probs.sum()) <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id


# ------------------------
# Fast incremental decoding loop
# ------------------------

def decode_with_multicheck_fast(
    model,
    tok,
    prompt_ids_1xT: torch.Tensor,
    stop_ids: set,
    reg,
    norm_mode: str,
    reg_layer: int,
    reg_thr: float,
    consecutive_n: int,
    insert_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    """
    Returns:
      full_ids_1xS (torch.LongTensor on same device as prompt)
      inserted (bool)
      insert_step (int or None)
      checkpoints (list of dicts)  # identical schema as original script
    """
    device = prompt_ids_1xT.device
    eos_id = tok.eos_token_id

    inserted = False
    insert_step = None
    checkpoints = []
    consec_hit = 0


    # If the model already emits </think>, NEVER insert another </think> (would corrupt reasoning format).
    close_ids = None
    if (insert_text or "") and ("</think>" in (insert_text or "")):
        try:
            close_ids = tok("</think>", add_special_tokens=False)["input_ids"]
            if not close_ids:
                close_ids = None
        except Exception:
            close_ids = None

    think_closed = False
    _close_match = 0  # subsequence match state for close_ids
    # ---------- Prefill (prompt) ----------
    with torch.inference_mode():
        out = model(prompt_ids_1xT, use_cache=True, output_hidden_states=False, return_dict=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]  # next token distribution

    generated = []  # list[int] of generated token ids (excluding prompt)
    remaining = int(max_new_tokens)

    # helper to append tokens to state and update logits/past
    def _step_feed(token_id: int, need_hidden: bool):
        nonlocal past
        tok_tensor = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
        with torch.inference_mode():
            o = model(
                tok_tensor,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=bool(need_hidden),
                return_dict=True,
            )
        past = o.past_key_values
        logits_next = o.logits[:, -1, :]
        hs = o.hidden_states if need_hidden else None
        return logits_next, hs


    def _update_think_closed(last_token_id: int):
        nonlocal think_closed, _close_match
        if think_closed or close_ids is None:
            return
        tid = int(last_token_id)
        # Simple streaming subsequence matcher
        if tid == int(close_ids[_close_match]):
            _close_match += 1
            if _close_match >= len(close_ids):
                think_closed = True
                return
        else:
            # restart match; allow immediate re-match if current token equals first
            _close_match = 1 if tid == int(close_ids[0]) else 0

    # ---------- Generate loop ----------
    # First sample based on prompt next_logits
    while remaining > 0:
        # sample next token
        token_id = sample_next_token(
            next_logits,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        generated.append(int(token_id))
        _update_think_closed(int(token_id))
        remaining -= 1

        # stop on EOS
        if eos_id is not None and int(token_id) == int(eos_id):
            break

        # Now we are about to feed THIS token to advance cache and get logits for next step.
        # If THIS token is a checkpoint token, this is the "current step" hidden we want.
        is_checkpoint = int(token_id) in stop_ids

        do_check = (not inserted) and (not think_closed) and (reg is not None) and is_checkpoint
        need_hidden = bool(do_check)

        next_logits, hs = _step_feed(token_id, need_hidden=need_hidden)

        if is_checkpoint:
            # Build checkpoint dict EXACTLY like original
            pos_full = int(prompt_ids_1xT.shape[-1] + len(generated) - 1)
            ck = {
                "step": len(checkpoints),
                "pos_full": pos_full,
                "last_token_id": int(token_id),
                "remaining_after": int(remaining),

                "reg_used": False,
                "reg_pred": None,
                "reg_norm_mode": norm_mode,
                "reg_threshold": float(reg_thr),

                "consecutive_hit": int(consec_hit),
                "do_insert": False,
                "reason": None,
            }

            do_insert = False

            if (not inserted) and (reg is not None):
                # hs is a tuple: (emb, layer1, ..., layerN) or similar; we align with original indexing semantics.
                if hs is None or len(hs) == 0:
                    consec_hit = 0
                    ck["reason"] = "reg_rep_none_reset_consec"
                    ck["consecutive_hit"] = int(consec_hit)
                else:
                    L = int(reg_layer)
                    if L < 0:
                        L = len(hs) - 1
                    if L >= len(hs):
                        L = len(hs) - 1
                    # For incremental step, seq_len=1, so take [0, -1, :]
                    h = hs[L]  # [1, 1, H]
                    rep = h[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32)

                    pred = safe_predict_regressor(reg, rep)
                    pred = maybe_clip_pred(pred, norm_mode)

                    hit = bool(pred < reg_thr)
                    if hit:
                        consec_hit += 1
                    else:
                        consec_hit = 0

                    do_insert = bool((not think_closed) and (consec_hit >= consecutive_n))

                    ck.update({
                        "reg_used": True,
                        "reg_pred": float(pred),
                        "consecutive_hit": int(consec_hit),
                        "do_insert": bool(do_insert),
                        "reason": "reg_ok",
                    })
            
                    if think_closed:
                        ck["do_insert"] = False
                        ck["reason"] = "think_already_closed_skip_insert"
            else:
                ck["reason"] = "no_reg_or_already_inserted"
                consec_hit = 0
                ck["consecutive_hit"] = int(consec_hit)

            checkpoints.append(ck)

            if do_insert and (not think_closed) and (insert_text or ""):
                # Insert immediately
                insert_ids = tok(
                    insert_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"][0].tolist()

                inserted = True
                insert_step = ck["step"]

                # Feed inserted tokens into cache (so tail generation continues correctly)
                for tid in insert_ids:
                    generated.append(int(tid))
                    remaining -= 1
                    if remaining < 0:
                        remaining = 0
                        break
                    # Stop if EOS appears inside inserted text (unlikely but safe)
                    if eos_id is not None and int(tid) == int(eos_id):
                        remaining = 0
                        break
                    # after insertion, we generate to end WITHOUT any checkpoint checks
                    next_logits, _ = _step_feed(int(tid), need_hidden=False)

                # Tail generation: no checkpoint logic
                while remaining > 0:
                    tid = sample_next_token(
                        next_logits,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                    )
                    generated.append(int(tid))
                    remaining -= 1
                    if eos_id is not None and int(tid) == int(eos_id):
                        remaining = 0
                        break
                    next_logits, _ = _step_feed(int(tid), need_hidden=False)

                break  # done

    full_ids = torch.cat(
        [prompt_ids_1xT, torch.tensor([generated], device=device, dtype=torch.long)],
        dim=1
    )
    return full_ids, inserted, insert_step, checkpoints


# ------------------------
# Worker
# ------------------------

def worker(rank, world_size, args):
    set_seeds(int(args.seed) + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "test.jsonl")
    data = read_jsonl(dataset_path)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    out_dir = os.path.join(args.output_path, model_name, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    shard_path = os.path.join(out_dir, f"results.shard{rank}.jsonl")
    done = load_done_indices(shard_path)

    model, tok = load_model_tokenizer(args.model_name_or_path, args.trust_remote_code, device)
    stop_ids = build_stop_ids_contains(tok, needle="ĊĊ")

    # -------- load sklearn regressor (CPU) --------
    reg = None
    meta = None
    if args.reg and args.reg_meta:
        reg = joblib.load(args.reg)
        with open(args.reg_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta_best_layer = int(meta["best_layer"]) if meta else -1
    meta_k = int(meta.get("k", 0)) if meta else 0

    norm_mode = "log"
    try:
        norm_mode = str(meta.get("target", {}).get("norm_mode", "log")) if meta else "log"
    except Exception:
        norm_mode = "log"

    reg_layer = int(args.reg_layer) if args.reg_layer is not None else meta_best_layer
    reg_thr = float(args.reg_threshold)
    consecutive_n = int(args.consecutive_n)

    indices = [i for i in range(len(data)) if (i % world_size) == rank]
    pbar = tqdm(total=len(indices), desc=f"rank {rank}", position=rank, leave=True)

    sys_prompt = args.system_prompt.strip() if args.system_prompt else "Please reason step by step, and put your final answer within \\boxed{}."

    for i in indices:
        if i in done:
            pbar.update(1)
            continue

        q = data[i]
        q_text = q.get("problem", "")

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q_text},
        ]

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_len0 = int(inputs["input_ids"].shape[-1])

        remaining_budget = int(args.max_new_tokens)
        full_ids = inputs["input_ids"]  # [1, T]
        inserted = False
        insert_step = None
        checkpoints = []

        try:
            full_ids, inserted, insert_step, checkpoints = decode_with_multicheck_fast(
                model=model,
                tok=tok,
                prompt_ids_1xT=full_ids,
                stop_ids=stop_ids,
                reg=reg if (reg is not None and meta is not None) else None,
                norm_mode=norm_mode,
                reg_layer=reg_layer,
                reg_thr=reg_thr,
                consecutive_n=consecutive_n,
                insert_text=args.insert_text,
                max_new_tokens=remaining_budget,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
        except torch.cuda.OutOfMemoryError:
            _clear_cuda()
            pbar.update(1)
            continue

        # decode final (keep identical to original)
        gen_all_ids = full_ids[0, prompt_len0:].detach().cpu()
        response_text = tok.decode(gen_all_ids, skip_special_tokens=True)
        generate_response_length = int(gen_all_ids.numel())

        out_row = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "generate_response_length": generate_response_length,

            "inserted": bool(inserted),
            "insert_step": insert_step,
            "insert_policy": "reg_multicheck_consecutive" if (reg is not None and meta is not None) else f"fallback_{args.fallback}",

            "reg_threshold": float(reg_thr),
            "consecutive_n": int(consecutive_n),
            "reg_layer": int(reg_layer),
            "reg_meta_k": int(meta_k),

            "checkpoints": checkpoints,
        }

        append_jsonl(shard_path, out_row)
        done.add(i)

        del full_ids, gen_all_ids, inputs
        _clear_cuda()
        pbar.update(1)

    pbar.close()


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)

    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=81920)

    ap.add_argument("--insert_text", type=str, default="\\n</think>\\n\\n")
    ap.add_argument("--hs_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])  # kept for CLI-compat
    ap.add_argument("--system_prompt", type=str, default="")

    # ---- regressor args ----
    ap.add_argument("--reg", type=str, default="", help="Path to remain_reg.joblib (sklearn Pipeline)")
    ap.add_argument("--reg_meta", type=str, default="", help="Path to remain_reg_meta.json")
    ap.add_argument("--reg_layer", type=int, default=None, help="Override meta.best_layer (default: use meta)")
    ap.add_argument("--reg_threshold", type=float, default=0.3, help="Trigger if pred < threshold")
    ap.add_argument("--consecutive_n", type=int, default=1, help="Need pred<thr consecutively N times to insert")

    # kept for compatibility; reg path usually uses never
    ap.add_argument("--fallback", type=str, default="never", choices=["random", "never", "always"])

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()

