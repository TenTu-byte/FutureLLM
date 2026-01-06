# -*- coding: utf-8 -*-
"""
train_remain_length_reg_auto_layer.py  (Ridge-only, 3 norm modes)

Train a regressor:
  f(hidden_at_double_newline[layer]) -> target
where target is:
  - raw   : remain_length (original scale)
  - minmax: min-max(remain_length) in [0,1]
  - log   : min-max(log1p(remain_length)) in [0,1]

Features:
- Supports auto layer selection: --layer -1
- Grid search over (layer × ridge_alpha) using train/val
- Final output: ONE regressor pipeline (joblib) + meta json

Expected hidden_*.pt format:
  obj["hs"] : dict[layer_id -> Tensor[K,H]]   (K events per sample, H hidden dim)
  obj["remain_length"] : list/array length K

[NEW]
- Optionally exclude "wrong ids" (so we only train on questions the model answered correctly).
  By default, if ./wrong_id.jsonl exists, we will exclude ids listed in it.
  We infer question id from hidden file name: hidden_<ID>.pt
"""

import os
import glob
import json
import time
import argparse
import re
import ast  # ✅ NEW: robust parsing fallback
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

import joblib


# -------------------------
# utils
# -------------------------

_HIDDEN_ID_RE = re.compile(r"^hidden_(\d+)\.pt$")


def parse_hidden_file_id(path: str) -> Optional[int]:
    """
    Infer question id from filename like hidden_123.pt -> 123.
    Return None if filename doesn't match.
    """
    base = os.path.basename(path)
    m = _HIDDEN_ID_RE.match(base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_wrong_ids_jsonl(path: str) -> Set[int]:
    """
    Robust loader for wrong id files.

    Supports:
      - JSONL: each line is a JSON object (or list)
      - JSON: whole file is a JSON object/list (possibly pretty-printed across lines)
      - Python-literal dict/list (single quotes), via ast.literal_eval fallback
      - UTF-8 BOM

    We collect/union all "ids" fields, or treat a list as ids directly.
    """
    wrong: Set[int] = set()
    if not path:
        return wrong
    if not os.path.exists(path):
        return wrong

    def _extract_ids(obj: Any) -> List[int]:
        ids = None
        if isinstance(obj, dict):
            ids = obj.get("ids", None)
        elif isinstance(obj, list):
            ids = obj

        out: List[int] = []
        if isinstance(ids, list):
            for x in ids:
                try:
                    out.append(int(x))
                except Exception:
                    pass
        return out

    # Read whole file once (handles JSON pretty-print, BOM, etc.)
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # First, try parse as whole-file JSON (common for .json, or pretty-printed)
    s_all = content.strip()
    if s_all:
        # Try strict JSON
        try:
            obj = json.loads(s_all)
            for i in _extract_ids(obj):
                wrong.add(i)
            return wrong
        except Exception:
            # Try python-literal (single quotes etc.)
            try:
                obj = ast.literal_eval(s_all)
                for i in _extract_ids(obj):
                    wrong.add(i)
                return wrong
            except Exception:
                # Fall back to JSONL line-by-line below
                pass

    # Fallback: JSONL line-by-line
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            # Skip comment-like lines if any
            if s.startswith("#") or s.startswith("//"):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                except Exception as e:
                    raise ValueError(f"JSON parse error in wrong_id file at line {ln}: {e}")

            for i in _extract_ids(obj):
                wrong.add(i)

    return wrong


def find_hidden_files_in_folder(folder: str) -> List[str]:
    pat1 = os.path.join(folder, "hidden_*.pt")
    pat2 = os.path.join(folder, "**", "hidden_*.pt")
    files = sorted(glob.glob(pat1)) + sorted(glob.glob(pat2, recursive=True))
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def gather_folders(args) -> List[str]:
    folders: List[str] = []
    if isinstance(args.folder, list):
        folders.extend([x for x in args.folder if x])
    if getattr(args, "folders", None):
        folders.extend([x for x in args.folders if x])

    for k in range(1, 33):
        v = getattr(args, f"folder{k}", "") or ""
        v = v.strip()
        if v:
            folders.append(v)

    seen = set()
    out = []
    for f in folders:
        f = f.strip()
        if not f or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(dtype=torch.float32).cpu().numpy()


def transform_target(remain: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    Return:
      y      : regression target (raw or in [0,1])
      meta   : transform metadata for possible inverse transform downstream
      label  : readable label

    mode:
      - raw
      - minmax
      - log   (log1p then minmax)
    """
    r = remain.astype(np.float32)

    if mode == "raw":
        meta = {"mode": "raw"}
        return r, meta, "remain_length (raw)"

    if mode == "log":
        x = np.log1p(r)
        label = "log1p(remain_length) min-max"
    else:
        x = r
        label = "remain_length min-max"

    mn = float(x.min())
    mx = float(x.max())
    meta = {"mode": mode, "y_min": mn, "y_max": mx}

    if mx - mn < 1e-12:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - mn) / (mx - mn)

    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y, meta, label


# -------------------------
# load points (event-level)
# -------------------------

def load_hidden_meta(hidden_path: str) -> Optional[Dict[str, Any]]:
    try:
        obj = torch.load(hidden_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    hs = obj.get("hs", None)
    rem = obj.get("remain_length", None)

    if not isinstance(hs, dict) or len(hs) == 0:
        return None
    if rem is None:
        return None

    remain = np.asarray(rem, dtype=np.float32)
    if remain.ndim != 1 or len(remain) <= 0:
        return None

    K = int(len(remain))

    layers = set()
    H = None
    for lid, mat in hs.items():
        try:
            lid_int = int(lid)
        except Exception:
            continue
        if torch.is_tensor(mat) and mat.ndim == 2 and int(mat.shape[0]) == K:
            if H is None:
                H = int(mat.shape[1])
                layers.add(lid_int)
            else:
                if int(mat.shape[1]) == H:
                    layers.add(lid_int)

    if H is None or len(layers) == 0:
        return None

    return {"K": K, "H": H, "layers": layers, "remain": remain}


def build_event_index(
    folders: List[str],
    k: int,
    max_points: int,
    seed: int,
    wrong_ids: Optional[Set[int]] = None,
    disable_wrong_filter: bool = False,
) -> Tuple[List[Tuple[str, int, str]], np.ndarray, int, List[int]]:
    rng = np.random.RandomState(seed)

    all_candidates: List[Tuple[str, int, str]] = []
    file_metas: Dict[str, Dict[str, Any]] = {}

    wrong_ids = wrong_ids or set()
    filtered_files = 0
    kept_files = 0
    unknown_id_files = 0

    for folder in folders:
        tag = os.path.basename(folder.rstrip("/\\")) or folder
        for hp in find_hidden_files_in_folder(folder):
            if (not disable_wrong_filter) and wrong_ids:
                hid = parse_hidden_file_id(hp)
                if hid is None:
                    unknown_id_files += 1
                else:
                    if hid in wrong_ids:
                        filtered_files += 1
                        continue

            meta = load_hidden_meta(hp)
            if meta is None:
                continue

            file_metas[hp] = meta
            kept_files += 1
            K = meta["K"]
            if k >= 0:
                if k < K:
                    all_candidates.append((hp, int(k), tag))
            else:
                for kk in range(K):
                    all_candidates.append((hp, int(kk), tag))

    if not all_candidates:
        raise RuntimeError(
            "No usable events found. Expect hidden_*.pt contains keys: 'hs' (dict[layer->Tensor[K,H]]) "
            "and 'remain_length' (len K). "
            "Also check your wrong_id filter didn't filter everything."
        )

    if (not disable_wrong_filter) and wrong_ids:
        print(
            f"[info] wrong-id filter ON | wrong_ids={len(wrong_ids)} "
            f"| filtered_files={filtered_files} | kept_files={kept_files} | unknown_id_files={unknown_id_files}"
        )

    if max_points and len(all_candidates) > int(max_points):
        idx = rng.choice(len(all_candidates), size=int(max_points), replace=False)
        idx = np.asarray(idx, dtype=np.int64)
        candidates = [all_candidates[i] for i in idx.tolist()]
    else:
        candidates = all_candidates

    # enforce consistent H across sampled files
    H = None
    used_files = sorted(set(hp for hp, _, _ in candidates))
    bad = set()
    for hp in used_files:
        meta = file_metas[hp]
        if H is None:
            H = int(meta["H"])
        elif int(meta["H"]) != int(H):
            bad.add(hp)
    if bad:
        candidates = [(hp, kk, tag) for (hp, kk, tag) in candidates if hp not in bad]
        used_files = sorted(set(hp for hp, _, _ in candidates))

    if not candidates:
        raise RuntimeError("After enforcing consistent hidden dim H, no events remain.")

    # common layers
    common_layers = None
    for hp in used_files:
        layers = set(file_metas[hp]["layers"])
        if common_layers is None:
            common_layers = layers
        else:
            common_layers &= layers
    if common_layers is None or len(common_layers) == 0:
        raise RuntimeError("Common layer set is empty across sampled files.")
    common_layers = sorted(int(x) for x in common_layers)

    # y raw
    y_rem = np.zeros((len(candidates),), dtype=np.float32)
    for i, (hp, kk, _) in enumerate(candidates):
        y_rem[i] = float(file_metas[hp]["remain"][kk])

    return candidates, y_rem, int(H), common_layers


def load_layer_matrix_for_events(
    events: List[Tuple[str, int, str]],
    layer_id: int,
    H: int,
) -> np.ndarray:
    X = np.zeros((len(events), H), dtype=np.float32)
    by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # hp -> [(row_i, kk)]
    for i, (hp, kk, _) in enumerate(events):
        by_file[hp].append((i, kk))

    for hp, pairs in by_file.items():
        obj = torch.load(hp, map_location="cpu")
        hs = obj["hs"]
        mat = hs[int(layer_id)]  # Tensor[K,H]
        for i, kk in pairs:
            X[i] = _to_numpy_f32(mat[kk])

    return X


# -------------------------
# ridge grid
# -------------------------

def build_ridge_grid(seed: int) -> List[Dict[str, Any]]:
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    grid: List[Dict[str, Any]] = []
    for a in alphas:
        grid.append({
            "name": "Ridge",
            "make": lambda aa=a: Ridge(alpha=float(aa), random_state=seed),
            "params": {"alpha": float(a)},
        })
    return grid


def make_pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def maybe_clip_pred(pred: np.ndarray, norm_mode: str) -> np.ndarray:
    if norm_mode in ("log", "minmax"):
        return np.clip(pred, 0.0, 1.0).astype(np.float32)
    return pred.astype(np.float32)


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--folder", type=str, action="append", default=[],
                    help="One or more folders. You can pass --folder multiple times.")
    ap.add_argument("--folders", type=str, nargs="+", default=[])
    for k in range(1, 33):
        ap.add_argument(f"--folder{k}", type=str, default="")

    ap.add_argument("--k", type=int, default=0,
                    help="Which '\\n\\n' event index to use. 0=first. -1=use ALL events.")
    ap.add_argument("--layer", type=int, required=True,
                    help="If -1, scan all common layers and pick best; otherwise use the specified layer.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size_in_train", type=float, default=0.25)

    ap.add_argument("--max_points", type=int, default=0,
                    help="Cap #events for training (after expanding by k). 0 means no cap.")

    ap.add_argument(
        "--norm_mode",
        type=str,
        default="log",
        choices=["raw", "log", "minmax"],
        help="Target transform: raw=use original remain_length; log=log1p then minmax; minmax=raw then minmax"
    )

    ap.add_argument("--out_dir", type=str, default="",
                    help="Save dir for remain_reg.joblib + meta.json. Default: <first_folder>/remain_reg")

    ap.add_argument(
        "--wrong_id_path",
        type=str,
        default="wrong_id.jsonl",
        help="Path to wrong_id.jsonl (jsonl/json). IDs inside will be excluded. Default: ./wrong_id.jsonl"
    )
    ap.add_argument(
        "--disable_wrong_filter",
        action="store_true",
        help="Disable excluding wrong ids even if wrong_id_path exists."
    )

    args = ap.parse_args()

    folders = gather_folders(args)
    if not folders:
        raise SystemExit("No folder provided. Use --folder / --folders / --folder1..")

    seed = int(args.seed)
    req_layer = int(args.layer)
    kk = int(args.k)
    norm_mode = str(args.norm_mode)

    # load wrong ids (if file exists and not disabled)
    wrong_ids: Set[int] = set()
    if not args.disable_wrong_filter:
        wrong_path = (args.wrong_id_path or "").strip()
        if wrong_path and os.path.exists(wrong_path):
            wrong_ids = load_wrong_ids_jsonl(wrong_path)
        else:
            print(f"[info] wrong-id filter: file not found at '{wrong_path}', skip filtering.")

    # 1) build event index (and common layers)
    events, y_rem_raw, H, common_layers = build_event_index(
        folders=folders,
        k=kk,
        max_points=int(args.max_points),
        seed=seed,
        wrong_ids=wrong_ids,
        disable_wrong_filter=bool(args.disable_wrong_filter),
    )
    N = len(events)
    folder_tags = np.asarray([tag for (_, _, tag) in events], dtype=object)

    # transform target
    y, y_meta, y_label = transform_target(y_rem_raw, mode=norm_mode)
    print(f"[info] events={N} | H={H} | common_layers={len(common_layers)} | k={kk} | target={y_label}")

    strat = folder_tags if len(set(folder_tags.tolist())) > 1 else None

    idx_all = np.arange(N, dtype=np.int64)
    idx_tr_all, idx_te = train_test_split(
        idx_all, test_size=float(args.test_size), random_state=seed, stratify=strat
    )
    idx_tr, idx_val = train_test_split(
        idx_tr_all,
        test_size=float(args.val_size_in_train),
        random_state=seed,
        stratify=(folder_tags[idx_tr_all] if strat is not None else None),
    )

    y_tr = y[idx_tr]
    y_val = y[idx_val]
    y_te = y[idx_te]

    ridge_grid = build_ridge_grid(seed=seed)

    # 2) search (layer × ridge_alpha)
    print("\n===== Grid search (layer × ridge_alpha) on train/val =====")
    best = None
    all_scores: List[Dict[str, Any]] = []

    layers_to_try = common_layers
    if req_layer != -1:
        if req_layer not in common_layers:
            raise RuntimeError(f"--layer {req_layer} not in common_layers. Example available: {common_layers[:10]} ...")
        layers_to_try = [int(req_layer)]
        print(f"[info] use provided layer={req_layer} (skip layer scan)")

    t0 = time.time()
    for lid in layers_to_try:
        X = load_layer_matrix_for_events(events, layer_id=int(lid), H=H)
        X_tr = X[idx_tr]
        X_val = X[idx_val]
        X_te = X[idx_te]  # report only

        for cfg in ridge_grid:
            pipe = make_pipeline(cfg["make"]())
            pipe.fit(X_tr, y_tr)

            pred_val = maybe_clip_pred(pipe.predict(X_val), norm_mode)
            val_r2 = float(r2_score(y_val, pred_val))
            val_mae = float(mean_absolute_error(y_val, pred_val))

            pred_te = maybe_clip_pred(pipe.predict(X_te), norm_mode)
            te_r2 = float(r2_score(y_te, pred_te))
            te_mae = float(mean_absolute_error(y_te, pred_te))

            rec = {
                "layer": int(lid),
                "regressor": "Ridge",
                "params": cfg["params"],
                "val_r2": val_r2,
                "val_mae": val_mae,
                "test_r2": te_r2,
                "test_mae": te_mae,
            }
            all_scores.append(rec)

            key = (val_r2, -val_mae)
            if best is None or key > (best["val_r2"], -best["val_mae"]):
                best = rec

        if (int(lid) % 4) == 0:
            layer_best = [r for r in all_scores if r["layer"] == int(lid)]
            layer_best = sorted(layer_best, key=lambda d: (d["val_r2"], -d["val_mae"]), reverse=True)[0]
            print(f"[scan] layer={int(lid):3d} | best_alpha={layer_best['params']['alpha']} val_R2={layer_best['val_r2']:.4f} val_MAE={layer_best['val_mae']:.4f}")

    if best is None:
        raise RuntimeError("Search failed: no model successfully trained.")

    t1 = time.time()
    print("\n===== Best config (by val_R2, tie-break val_MAE) =====")
    print(f"best_layer : {best['layer']}")
    print(f"best_alpha : {best['params']['alpha']}")
    print(f"val_R2/MAE : {best['val_r2']:.4f} / {best['val_mae']:.4f}")
    print(f"test_R2/MAE: {best['test_r2']:.4f} / {best['test_mae']:.4f}")
    print(f"[time] search_time={t1 - t0:.2f}s | tried={len(all_scores)}")

    # 3) refit best on train_all
    best_layer = int(best["layer"])
    best_alpha = float(best["params"]["alpha"])

    X_best = load_layer_matrix_for_events(events, layer_id=best_layer, H=H)
    X_tr_all = X_best[idx_tr_all]
    X_te = X_best[idx_te]
    y_tr_all = y[idx_tr_all]

    final_pipe = make_pipeline(Ridge(alpha=best_alpha, random_state=seed))
    final_pipe.fit(X_tr_all, y_tr_all)

    pred_te = maybe_clip_pred(final_pipe.predict(X_te), norm_mode)
    te_r2 = float(r2_score(y_te, pred_te))
    te_mae = float(mean_absolute_error(y_te, pred_te))

    print("\n===== Final refit on train_all =====")
    print(f"test_R2/MAE: {te_r2:.4f} / {te_mae:.4f}")

    # 4) save
    out_dir = (args.out_dir.strip() or os.path.join(folders[0], "remain_reg"))
    os.makedirs(out_dir, exist_ok=True)

    joblib_path = os.path.join(out_dir, "remain_reg.joblib")
    meta_path = os.path.join(out_dir, "remain_reg_meta.json")

    joblib.dump(final_pipe, joblib_path)

    scored_sorted = sorted(all_scores, key=lambda d: (d["val_r2"], -d["val_mae"]), reverse=True)

    meta = {
        "task": "remain_length_regression",
        "model_family": "RidgeOnly",
        "best_layer": best_layer,
        "k": int(kk),
        "target": {
            "what": "remain_length" if norm_mode == "raw" else "normalized_remain_length",
            "norm_mode": norm_mode,
            "norm_label": y_label,
            "norm_meta": y_meta,
        },
        "selection": {
            "criterion": "max(val_R2), tie-break min(val_MAE)",
            "best_alpha": best_alpha,
            "val": {"r2": float(best["val_r2"]), "mae": float(best["val_mae"])},
            "test_quick_during_search": {"r2": float(best["test_r2"]), "mae": float(best["test_mae"])},
            "test_after_refit_on_train_all": {"r2": float(te_r2), "mae": float(te_mae)},
        },
        "grid": {
            "layers_tried": layers_to_try,
            "alphas_tried": [float(r["params"]["alpha"]) for r in ridge_grid],
            "topk": scored_sorted[:50],
        },
        "data": {
            "N_events": int(N),
            "H": int(H),
            "common_layers": common_layers,
            "folders": folders,
            "max_points": int(args.max_points),
            "split": {
                "test_size": float(args.test_size),
                "val_size_in_train": float(args.val_size_in_train),
                "N_train_all": int(len(idx_tr_all)),
                "N_train": int(len(idx_tr)),
                "N_val": int(len(idx_val)),
                "N_test": int(len(idx_te)),
            },
        },
        "wrong_id_filter": {
            "enabled": (not bool(args.disable_wrong_filter)) and bool(wrong_ids),
            "wrong_id_path": (args.wrong_id_path or ""),
            "num_wrong_ids": int(len(wrong_ids)),
            "note": "Exclude hidden_<ID>.pt if ID in wrong_ids; files with non-matching filename are kept.",
        },
        "timestamp": time.time(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved regressor: {joblib_path}")
    print(f"[OK] saved meta     : {meta_path}")


if __name__ == "__main__":
    main()



