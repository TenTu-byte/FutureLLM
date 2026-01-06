import json
from transformers import AutoTokenizer

import re
import importlib.util
import os
import argparse

import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
import pdb


def parse_list(arg):
    return arg.split(',')


def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)


def _default_correctness_path(generation_path: str) -> str:
    """Default correctness jsonl path next to generation_path."""
    gen_dir = os.path.dirname(os.path.abspath(generation_path))
    base = os.path.basename(generation_path)
    base_noext = os.path.splitext(base)[0] if base else "generations"
    return os.path.join(gen_dir, f"{base_noext}.correctness.jsonl")


def _write_jsonl_line(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./Data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--generation_path", default="test", type=str)

    parser.add_argument("--prompt_type", default="qwen-base", type=str)

    # NEW: save per-sample correctness to a separate jsonl
    parser.add_argument(
        "--correctness_jsonl",
        type=str,
        default="",
        help=(
            "Save per-sample correctness records to this jsonl. "
            "If empty, will save next to generation_path with suffix .correctness.jsonl"
        ),
    )

    args = parser.parse_args()
    return args


def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")

    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")

    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def infer(args):
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)

    print("llm generate done")
    print(len(file_outputs))

    # NEW: correctness writer (for infer as well)
    corr_path = args.correctness_jsonl or _default_correctness_path(args.generation_path)
    os.makedirs(os.path.dirname(os.path.abspath(corr_path)), exist_ok=True)
    corr_f = open(corr_path, "w", encoding="utf-8")

    pass_at_k_list = []
    k = args.k

    correct_cnt = 0
    try:
        for i in tqdm(range(len(file_outputs)), "check correct..."):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_responses = file_outputs[i]['generated_responses']

            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)

            if is_correct:
                correct_cnt += 1

            file_outputs[i]['generated_answers'] = generated_answers
            file_outputs[i]['gold_answer'] = gt_ans
            file_outputs[i]['is_correct'] = is_correct
            file_outputs[i]['answers_correctness'] = is_correct_list

            # NEW: per-sample correctness record (pass@1 = first sample)
            qid = d.get('id', i) if isinstance(d, dict) else i
            pred1 = generated_answers[0] if generated_answers else None
            correct1 = bool(is_correct_list[0]) if is_correct_list else False
            rec = {
                "idx": i,
                "id": qid,
                "gold_answer": gt_ans,
                "pred_answer": pred1,
                "is_correct": correct1,                 # pass@1 口径（第一个生成）
                "answers_correctness": is_correct_list,  # 兼容多采样（可选）
            }
            _write_jsonl_line(corr_f, rec)

            if len(is_correct_list) > 1:
                correct_answers = sum(is_correct_list)
                n = len(generated_answers)
                if correct_answers > 0:
                    if n - correct_answers < k:
                        pass_at_k = 1
                    else:
                        pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                    pass_at_k_list.append(pass_at_k)
                else:
                    pass_at_k_list.append(0)

    finally:
        corr_f.close()

    print(f"[INFO] Correctness jsonl saved to: {corr_path}")
    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")

    response_length = []
    token_num = []
    wait_num = []
    alt_num = []

    test_num = len(file_outputs)
    correct_num = 0
    for data in file_outputs:
        response_length.append(len(data['generated_responses'][0].split()))
        tokens_response_len = len(tokenizer(data['generated_responses'][0])['input_ids'])
        token_num.append(tokens_response_len)

    avg_response_length = sum(response_length) / test_num
    avg_token_num = sum(token_num) / test_num

    print("length:", avg_response_length)
    print('token_num:', avg_token_num)


def infer_2(args):
    import os, json, re
    from math import comb

    # -------- helpers --------
    def _extract_text_from_generated(g):
        """兼容 str/dict/list 的生成结果，尽量取到真正文本。"""
        if isinstance(g, str):
            return g
        if isinstance(g, dict):
            for key in ("text", "content", "generated_response", "generated_text",
                        "output", "output_text", "message", "response"):
                v = g.get(key, None)
                if isinstance(v, str):
                    return v
        if isinstance(g, list) and g:
            for item in g:
                t = _extract_text_from_generated(item)
                if t:
                    return t
        return ""

    def _get_response_text(data):
        """优先按你原结构取文本；否则兼容 outputs[0].outputs[0].text 结构。"""
        if isinstance(data, dict) and data.get('generated_responses'):
            first_gen = data['generated_responses'][0]
            return _extract_text_from_generated(first_gen)
        if isinstance(data, dict) and data.get('outputs'):
            o0 = data['outputs'][0] if data['outputs'] else None
            if isinstance(o0, dict) and o0.get('outputs'):
                oo0 = o0['outputs'][0]
                if isinstance(oo0, dict) and isinstance(oo0.get('text'), str):
                    return oo0['text']
        return ""

    # -------- load --------
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)

    print("llm generate done")
    print(len(file_outputs))

    # -------- NEW: open correctness writer --------
    corr_path = args.correctness_jsonl or _default_correctness_path(args.generation_path)
    os.makedirs(os.path.dirname(os.path.abspath(corr_path)), exist_ok=True)
    corr_f = open(corr_path, "w", encoding="utf-8")

    # -------- correctness & pass@k --------
    pass_at_k_list = []
    avg_at_k_list = []   # 新增：存每道题的 avg@k
    k = args.k

    correct_cnt = 0
    wrong_ids = []  # 记录答错题的 id
    # -------- 新增：按 batch 聚合的统计，用于 batch 逻辑的准确率方差 --------
    batch_correct_counts = [0] * k   # 每个 batch 在所有题上的「答对题数」
    batch_total_counts = [0] * k     # 每个 batch 实际参与的题目数（一般就是总题数）

    try:
        for i in tqdm(range(len(file_outputs)), "check correct..."):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)

            generated_responses = file_outputs[i]['generated_responses']
            generated_answers = [extract_answer(gr, args.data_name) for gr in generated_responses]
            is_correct_list = [check_is_correct(ga, gt_ans) for ga in generated_answers]
            is_correct = any(is_correct_list)

            if is_correct:
                correct_cnt += 1
            else:
                qid = d.get('id', i) if isinstance(d, dict) else i
                wrong_ids.append(qid)

            file_outputs[i]['generated_answers'] = generated_answers
            file_outputs[i]['gold_answer'] = gt_ans
            file_outputs[i]['is_correct'] = is_correct
            file_outputs[i]['answers_correctness'] = is_correct_list

            # -------- NEW: write per-sample correctness jsonl (pass@1) --------
            qid = d.get('id', i) if isinstance(d, dict) else i
            pred1 = generated_answers[0] if generated_answers else None
            correct1 = bool(is_correct_list[0]) if is_correct_list else False
            rec = {
                "idx": i,
                "id": qid,
                "gold_answer": gt_ans,
                "pred_answer": pred1,
                "is_correct": correct1,                 # pass@1: 只看第一个生成
                "answers_correctness": is_correct_list,  # 兼容多采样
            }
            _write_jsonl_line(corr_f, rec)

            # ---------- 新增：每个样本集的 avg@k ----------
            n_samples = len(is_correct_list)
            k_eff = min(k, n_samples) if n_samples > 0 else 0
            if k_eff > 0:
                avg_k_i = sum(is_correct_list[:k_eff]) / k_eff
            else:
                avg_k_i = 0.0
            avg_at_k_list.append(avg_k_i)

            # ---------- 新增：按 batch 聚合的统计 ----------
            # 对于第 j 个 batch，就看 is_correct_list[j] 是否为 True
            for j in range(k_eff):
                if is_correct_list[j]:
                    batch_correct_counts[j] += 1
                batch_total_counts[j] += 1

            if len(is_correct_list) > 1:
                correct_answers = sum(is_correct_list)
                n = len(generated_answers)
                if correct_answers > 0:
                    if n - correct_answers < k:
                        pass_at_k = 1
                    else:
                        pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                    pass_at_k_list.append(pass_at_k)
                else:
                    pass_at_k_list.append(0)

    finally:
        corr_f.close()

    print(f"[INFO] Correctness jsonl saved to: {corr_path}")

    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")

    # ---------- 新：按 batch 维度的 avg@k（准确率）方差 ----------
    # 对每个 batch j，计算它在所有题目上的准确率 acc_j
    batch_acc = []
    for j in range(k):
        if batch_total_counts[j] > 0:
            batch_acc.append(batch_correct_counts[j] / batch_total_counts[j])

    if batch_acc:
        n_b = len(batch_acc)

        mean_batch_acc = sum(batch_acc) / n_b
        var_batch_acc = sum((x - mean_batch_acc) ** 2 for x in batch_acc) / n_b  # 总体方差
        sd_batch_acc = var_batch_acc ** 0.5

        print(f"Batch acc list (k={n_b}): {batch_acc}")
        print(f"Avg accuracy over {n_b} batches: {mean_batch_acc:.4f}")
        print(f"Accuracy variance across batches: {var_batch_acc:.6f}")
        print(f"Accuracy SD across batches: {sd_batch_acc:.6f}")

    # （可选）如果你还想保留原来的“按题目聚合”的 avg@k 统计，也可以顺便打印：
    if avg_at_k_list:
        n_q = len(avg_at_k_list)
        mean_avg_k_q = sum(avg_at_k_list) / n_q
        var_avg_k_q = sum((x - mean_avg_k_q) ** 2 for x in avg_at_k_list) / n_q
        sd_avg_k_q = var_avg_k_q ** 0.5

        print(f"[Per-question avg@{k}] mean={mean_avg_k_q:.4f}, var={var_avg_k_q:.6f}, sd={sd_avg_k_q:.6f}")
    # ------------------------------------------------

    # -------- 保存错题 ID 到 JSON（不刷屏打印具体列表） --------
    out_dir = os.path.dirname(os.path.abspath(getattr(args, "generation_path", "wrong_ids.json")))
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"wrong_ids_{getattr(args, 'data_name', 'dataset')}_{getattr(args, 'split', 'split')}.json"
    out_path = os.path.join(out_dir, out_name)
    wrong_payload = {
        "data_name": getattr(args, "data_name", None),
        "split": getattr(args, "split", None),
        "count": len(wrong_ids),
        "ids": wrong_ids,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrong_payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrong IDs saved to: {out_path} (count={len(wrong_ids)})")

    # -------- token stats (全文 & think 段；按 avg@k 口径统计) --------
    # 每个样本一条：存「前 k 个生成的平均长度」
    # 按 batch 聚合的累积量：长度为 k
    batch_word_sum = [0.0] * k           # 每个 batch 的词数总和
    batch_token_sum = [0.0] * k          # 每个 batch 的 token 总和
    batch_think_token_sum = [0.0] * k    # 每个 batch 的 think 段 token 总和
    batch_count = [0] * k                # 每个 batch 参与的样本数

    think_found = 0       # 至少有一个生成出现 </think> 的题目数
    fallback_full = 0     # 所有生成都没出现 </think>，只能用全文兜底的题目数

    test_num = len(file_outputs)

    for data in file_outputs:
        # 优先用 generated_responses 这个结构
        gens = None
        if isinstance(data, dict) and "generated_responses" in data:
            gens = data["generated_responses"]

        sample_has_text = False
        sample_has_think = False

        if isinstance(gens, list) and len(gens) > 0:
            n_samples = len(gens)
            k_eff = min(k, n_samples)

            for j in range(k_eff):
                text = _extract_text_from_generated(gens[j])
                if not text:
                    continue

                sample_has_text = True

                # 词数
                wlen = len(text.split())
                batch_word_sum[j] += wlen

                # 整段 token 数
                tlen = len(tokenizer(text)["input_ids"])
                batch_token_sum[j] += tlen

                # —— 以 </think> 边界截断；找不到则取全文 ——
                lower = text.lower()
                idx = lower.find("</think>")
                if idx == -1:
                    idx = lower.find("&lt;/think&gt;")

                if idx != -1:
                    think_text = text[:idx]
                    sample_has_think = True
                else:
                    think_text = text

                t_think = len(
                    tokenizer(think_text, add_special_tokens=False)["input_ids"]
                ) if think_text else 0
                batch_think_token_sum[j] += t_think

                batch_count[j] += 1

        else:
            # 兼容老结构：退回到只看一条，当作 batch 0
            text = _get_response_text(data)
            if text:
                sample_has_text = True

                wlen = len(text.split())
                tlen = len(tokenizer(text)["input_ids"])

                lower = text.lower()
                idx = lower.find("</think>")
                if idx == -1:
                    idx = lower.find("&lt;/think&gt;")

                if idx != -1:
                    think_text = text[:idx]
                    sample_has_think = True
                else:
                    think_text = text

                t_think = len(
                    tokenizer(think_text, add_special_tokens=False)["input_ids"]
                ) if think_text else 0

                batch_word_sum[0] += wlen
                batch_token_sum[0] += tlen
                batch_think_token_sum[0] += t_think
                batch_count[0] += 1

        # 用于统计“有无 think 块”的题目数
        if sample_has_text:
            if sample_has_think:
                think_found += 1
            else:
                fallback_full += 1

    # 先得到每个 batch 的平均长度（在题目维度上求均值）
    batch_mean_len = []
    batch_mean_token = []
    batch_mean_think_token = []

    for j in range(k):
        if batch_count[j] > 0:
            batch_mean_len.append(batch_word_sum[j] / batch_count[j])
            batch_mean_token.append(batch_token_sum[j] / batch_count[j])
            batch_mean_think_token.append(batch_think_token_sum[j] / batch_count[j])

    def _mean_var(xs):
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)  # 总体方差
        return m, v

    # 再在 batch 维度上，对这 k 个 batch 的均值求「均值 + 方差」
    avg_response_length, var_response_length = _mean_var(batch_mean_len)
    avg_token_num, var_token_num = _mean_var(batch_mean_token)
    avg_think_token_num, var_think_token_num = _mean_var(batch_mean_think_token)
    print(batch_mean_token)
    print("length:", avg_response_length, "var:", var_response_length)
    print("token_num:", avg_token_num, "var:", var_token_num)
    print("sd", var_token_num**0.5)
    print("think_token_num:", avg_think_token_num, "var:", var_think_token_num)
    print(
        f"think blocks found by </think>: {think_found}/{test_num} (fallback_full={fallback_full})"
    )


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    infer_2(args)