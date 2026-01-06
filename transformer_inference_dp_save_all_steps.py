# -*- coding: utf-8 -*-

import os
import json
import argparse
import gc
import random
import numpy as np
import torch
import torch.multiprocessing as mp

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
# Core helpers
# ------------------------

def _find_all_subseq_positions(seq, subseq):
    """
    Return all start indices where subseq matches seq (allow overlaps).
    seq, subseq: list[int]
    """
    if not subseq:
        return []
    L, M = len(seq), len(subseq)
    out = []
    for s in range(0, L - M + 1):
        if seq[s:s + M] == subseq:
            out.append(s)
    return out


def build_stop_ids_contains(tok, needle: str = "ĊĊ"):
    """
    Build a set of token ids whose token string contains `needle`.
    For Qwen2/Qwen3 TokenizerFast:
      "\\n\\n"  -> 'ĊĊ'
      " \\n\\n" -> 'ĠĊĊ'
    Both contain 'ĊĊ'.
    """
    vocab = tok.get_vocab()  # token_str -> id
    stop_ids = {int(i) for s, i in vocab.items() if needle in s}
    if not stop_ids:
        raise RuntimeError(f'No tokens containing "{needle}" found in tokenizer vocab.')
    return stop_ids


def first_index_in_set(seq, id_set):
    """
    Return first index t where seq[t] in id_set, else None.
    seq: List[int]
    """
    s = set(int(x) for x in id_set)
    for t, tid in enumerate(seq):
        if int(tid) in s:
            return t
    return None


def _encode_no_special(tok, text: str):
    try:
        ids = tok.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]
    except Exception:
        return []


def _find_earliest_think_end(gen_ids_list, tokenizer):
    """
    Find earliest start index of a '</think>'-like marker in generated token ids.
    Return (think_end_gen, matched_pattern_ids). If not found, return (len(gen), []).
    """
    patterns = [
        "</think>",
        " </think>",
        "\n</think>",
        "\n\n</think>",
    ]
    best = None
    best_pat = []
    for p in patterns:
        pat_ids = _encode_no_special(tokenizer, p)
        if not pat_ids:
            continue
        pos = _find_all_subseq_positions(gen_ids_list, pat_ids)
        if not pos:
            continue
        start = int(pos[0])
        if best is None or start < best:
            best = start
            best_pat = pat_ids
    if best is None:
        return len(gen_ids_list), []
    return best, best_pat


# ------------------------
# Old function (kept for compatibility/reference)
# ------------------------

def save_double_newline_token_hs_first_only(
    model,
    tokenizer,
    full_ids_1xT: torch.Tensor,   # [1, T] on device
    prompt_len: int,
    save_path: str,
    hs_device: str = "auto",
):
    """
    Save all-layer hidden states at ONLY the FIRST occurrence of a token whose
    token string contains "ĊĊ" (e.g., 'ĊĊ' or 'ĠĊĊ') in generated tokens.
    """
    import os as _os

    stop_ids = build_stop_ids_contains(tokenizer, needle="ĊĊ")

    gen_ids_list = full_ids_1xT[0, prompt_len:].detach().cpu().tolist()
    t0 = first_index_in_set(gen_ids_list, stop_ids)

    pos_full = []
    hit_id = None
    if t0 is not None:
        hit_id = int(gen_ids_list[t0])
        pos_full = [prompt_len + t0]

    hs_out = {}

    def _forward_on(device):
        with torch.inference_mode():
            out = model(full_ids_1xT.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        if len(pos_full) == 0:
            return
        pos = torch.tensor(pos_full, dtype=torch.long, device=out.hidden_states[0].device)
        for lid, h in enumerate(out.hidden_states):  # each: [1, T, H]
            hs_out[int(lid)] = h[0].index_select(0, pos).to("cpu", non_blocking=True)  # [1, H]

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ("auto", "cuda") and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != "cpu":
                model.to("cpu")
            _forward_on(torch.device("cpu"))
    except torch.cuda.OutOfMemoryError:
        if hs_device == "cuda":
            raise
        _clear_cuda()
        model.to("cpu")
        _forward_on(torch.device("cpu"))
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    torch.save({"pos_full": pos_full, "pattern_ids": ([] if hit_id is None else [hit_id]), "hs": hs_out}, save_path)


# ------------------------
# NEW: save ALL "\n\n" token hidden states BEFORE </think>
# ------------------------

def save_double_newline_token_hs_before_think_all(
    model,
    tokenizer,
    full_ids_cpu_1xT: torch.Tensor,   # [1, T] on CPU (recommended)
    prompt_len: int,
    save_path: str,
    hs_device: str = "auto",
    block_size: int = 256,
):
    """
    Save all-layer hidden states at ALL occurrences of a token whose token string contains "ĊĊ"
    (e.g., 'ĊĊ' or 'ĠĊĊ') BUT ONLY those occurrences strictly BEFORE the first '</think>' marker.

    Also save remain_length list aligned 1-to-1 with every '\n\n' hit:
      remain_length[k] = (#tokens remaining in the thinking part, i.e., until </think>) after that '\n\n' token.

    File content:
      {
        "pos_gen": [int]*K,          # positions in generated tokens (0-based)
        "pos_full": [int]*K,         # positions in full sequence (prompt+gen)
        "pattern_ids": [int]*K,      # the hit token id at each position
        "think_end_gen": int,        # where </think> starts in generated tokens (or len(gen) if not found)
        "think_end_full": int,       # prompt_len + think_end_gen
        "think_pattern_ids": [int],  # which pattern ids matched (maybe empty)
        "remain_length": [int]*K,    # aligned with pos_gen / pos_full
        "hs": {layer_id: Tensor[K,H]}
      }
    """
    import os as _os

    stop_ids = build_stop_ids_contains(tokenizer, needle="ĊĊ")

    # work in python lists on CPU
    full_ids_list = full_ids_cpu_1xT[0].tolist()
    gen_ids_list = full_ids_list[prompt_len:]
    gen_len = len(gen_ids_list)

    think_end_gen, think_pat_ids = _find_earliest_think_end(gen_ids_list, tokenizer)  # start index of </think>
    think_end_gen = int(min(max(think_end_gen, 0), gen_len))
    think_end_full = prompt_len + think_end_gen

    # collect all hits before </think>
    pos_gen = []
    pos_full = []
    hit_token_ids = []
    remain_length = []
    for t, tid in enumerate(gen_ids_list):
        if t >= think_end_gen:
            break
        if int(tid) in stop_ids:
            pos_gen.append(int(t))
            pos_full.append(int(prompt_len + t))
            hit_token_ids.append(int(tid))
            # remaining tokens in thinking part AFTER this '\n\n' token
            remain_length.append(int(think_end_gen - (t + 1)))

    K = len(pos_full)

    # If no hits, still write a file for consistency
    hs_out = {}
    if K == 0:
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "pos_gen": pos_gen,
                "pos_full": pos_full,
                "pattern_ids": hit_token_ids,
                "think_end_gen": think_end_gen,
                "think_end_full": think_end_full,
                "think_pattern_ids": think_pat_ids,
                "remain_length": remain_length,
                "hs": hs_out,
            },
            save_path,
        )
        return

    # Decide compute device
    orig_device = next(model.parameters()).device
    if hs_device in ("auto", "cuda") and torch.cuda.is_available():
        compute_device = orig_device
    else:
        compute_device = torch.device("cpu")

    # Move model if needed
    moved = False
    if next(model.parameters()).device != compute_device:
        model.to(compute_device)
        moved = True

    # We do cache+chunk forward to avoid building full hidden states for the whole sequence
    # Gather hidden states only at pos_full.
    # Allocate per-layer list[K] then stack -> [K,H]
    per_layer = None  # will become List[List[Tensor]]
    pos_full_sorted = pos_full  # already increasing by construction

    try:
        with torch.inference_mode():
            T = len(full_ids_list)
            past = None
            cur = 0  # pointer in pos_full_sorted

            # to avoid repeated CPU->GPU copies, we slice from CPU tensor and move chunk to device
            for start in range(0, T, max(1, int(block_size))):
                end = min(T, start + max(1, int(block_size)))
                chunk = full_ids_cpu_1xT[:, start:end].to(compute_device, non_blocking=True)

                if past is None:
                    out = model(
                        input_ids=chunk,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    out = model(
                        input_ids=chunk,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                past = out.past_key_values

                if per_layer is None:
                    # out.hidden_states length = num_layers+1 (emb + each layer)
                    per_layer = [[] for _ in range(len(out.hidden_states))]

                # consume all target positions that fall into [start, end)
                while cur < K and pos_full_sorted[cur] < end:
                    p = pos_full_sorted[cur]
                    if p >= start:
                        local = int(p - start)  # index in this chunk
                        # each hidden: [1, chunk_len, H] -> take [0, local]
                        for lid, h in enumerate(out.hidden_states):
                            per_layer[lid].append(h[0, local].detach().to("cpu", non_blocking=True))
                    cur += 1

                # early stop if already collected all
                if cur >= K:
                    break

        # finalize
        if per_layer is not None:
            for lid, vecs in enumerate(per_layer):
                if len(vecs) != K:
                    # safety: if something went wrong, keep only what we have
                    if len(vecs) == 0:
                        continue
                hs_out[int(lid)] = torch.stack(vecs, dim=0)  # [K,H]

    except TypeError:
        # Fallback (less memory safe): full forward then index_select.
        # Keep it as a compatibility fallback for weird remote-code models.
        with torch.inference_mode():
            out = model(full_ids_cpu_1xT.to(compute_device), output_hidden_states=True, use_cache=False, return_dict=True)
        pos = torch.tensor(pos_full_sorted, dtype=torch.long, device=out.hidden_states[0].device)
        for lid, h in enumerate(out.hidden_states):
            hs_out[int(lid)] = h[0].index_select(0, pos).to("cpu", non_blocking=True)

    except torch.cuda.OutOfMemoryError:
        if hs_device == "cuda":
            raise
        _clear_cuda()
        model.to("cpu")
        moved = True
        compute_device = torch.device("cpu")
        # retry on CPU (chunk)
        hs_out = {}
        per_layer = None
        with torch.inference_mode():
            T = len(full_ids_list)
            past = None
            cur = 0
            for start in range(0, T, max(1, int(block_size))):
                end = min(T, start + max(1, int(block_size)))
                chunk = full_ids_cpu_1xT[:, start:end].to(compute_device)

                if past is None:
                    out = model(input_ids=chunk, use_cache=True, output_hidden_states=True, return_dict=True)
                else:
                    out = model(
                        input_ids=chunk,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                past = out.past_key_values

                if per_layer is None:
                    per_layer = [[] for _ in range(len(out.hidden_states))]

                while cur < K and pos_full_sorted[cur] < end:
                    p = pos_full_sorted[cur]
                    if p >= start:
                        local = int(p - start)
                        for lid, h in enumerate(out.hidden_states):
                            per_layer[lid].append(h[0, local].detach().to("cpu"))
                    cur += 1
                if cur >= K:
                    break

        if per_layer is not None:
            for lid, vecs in enumerate(per_layer):
                if len(vecs) == K:
                    hs_out[int(lid)] = torch.stack(vecs, dim=0)

    finally:
        # restore model device
        if moved and next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "pos_gen": pos_gen,
            "pos_full": pos_full,
            "pattern_ids": hit_token_ids,
            "think_end_gen": think_end_gen,
            "think_end_full": think_end_full,
            "think_pattern_ids": think_pat_ids,
            "remain_length": remain_length,
            "hs": hs_out,
        },
        save_path,
    )


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


def print_double_newline_token_info_once(tok, rank: int):
    """
    Print how "\\n\\n" is tokenized: ids + token strings.
    """
    try:
        pattern_ids = tok.encode("\n\n", add_special_tokens=False)
        pattern_ids = [int(x) for x in pattern_ids]
        pattern_toks = tok.convert_ids_to_tokens(pattern_ids)
        print(f"[rank {rank}] tokenizer('\\n\\n') pattern_ids = {pattern_ids}")
        print(f"[rank {rank}] tokenizer('\\n\\n') pattern_tokens = {pattern_toks}")
    except Exception as e:
        print(f"[rank {rank}] failed to print tokenizer('\\n\\n') info: {e}")


# ------------------------
# Worker
# ------------------------

def worker(rank, world_size, args):
    set_seeds(42 + rank)

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

    indices = [i for i in range(len(data)) if (i % world_size) == rank]
    pbar = tqdm(total=len(indices), desc=f"rank {rank}", position=rank, leave=True)

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

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

        # build generate kwargs (min_p may be unsupported depending on transformers)
        gen_kwargs = dict(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
        if args.min_p is not None:
            gen_kwargs["min_p"] = args.min_p

        try:
            with torch.inference_mode():
                try:
                    gen = model.generate(**inputs, **gen_kwargs)
                except TypeError:
                    # fallback if min_p is not supported
                    if "min_p" in gen_kwargs:
                        gen_kwargs.pop("min_p", None)
                        gen = model.generate(**inputs, **gen_kwargs)
                    else:
                        raise
        except torch.cuda.OutOfMemoryError:
            _clear_cuda()
            pbar.update(1)
            continue

        full_ids_1xT = gen.sequences  # [1, T] on device
        prompt_len = int(inputs["input_ids"].shape[-1])

        # decode response
        gen_ids = full_ids_1xT[0, prompt_len:].detach().cpu()
        response_text = tok.decode(gen_ids, skip_special_tokens=True)
        generate_response_length = int(gen_ids.numel())

        # ✅ NEW: save hidden_{i}.pt (ALL "\n\n" before </think>)
        if args.save_step_hs:
            hidden_path = os.path.join(out_dir, f"hidden_{i}.pt")
            try:
                # move ids to cpu to reduce gpu memory pressure
                full_ids_cpu_1xT = full_ids_1xT.detach().cpu()
                save_double_newline_token_hs_before_think_all(
                    model=model,
                    tokenizer=tok,
                    full_ids_cpu_1xT=full_ids_cpu_1xT,
                    prompt_len=prompt_len,
                    save_path=hidden_path,
                    hs_device=args.hs_device,
                    block_size=args.hs_block_size,
                )
                del full_ids_cpu_1xT
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                # still write jsonl even if hidden dump fails
                pass

        # write jsonl (only add generate_response_length)
        out_row = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "generate_response_length": generate_response_length,
        }
        append_jsonl(shard_path, out_row)
        done.add(i)

        del gen, inputs, full_ids_1xT, gen_ids
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

    # ✅ defaults aligned to author settings
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=81920)##--max_new_tokens=32768，16384

    ap.add_argument("--save_step_hs", action="store_true")
    ap.add_argument("--hs_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--hs_block_size", type=int, default=256)

    args = ap.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()

