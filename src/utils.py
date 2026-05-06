import os
import json
import math
import random
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ndcg_at_k(topk_items: np.ndarray, target_item: int) -> float:
    pos = np.where(topk_items == target_item)[0]
    if len(pos) == 0:
        return 0.0
    rank0 = int(pos[0])
    return 1.0 / math.log2(rank0 + 2.0)


def recall_at_k(topk_items: np.ndarray, target_item: int) -> float:
    return 1.0 if np.any(topk_items == target_item) else 0.0


def stable_desc_order(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    item_ids = np.arange(len(s), dtype=np.int64)
    order = np.lexsort((item_ids, -s))
    return order.astype(np.int32)


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    order = stable_desc_order(scores)
    if k >= len(order):
        return order
    return order[:k].astype(np.int32)


def stable_argmax_with_mask(scores: np.ndarray, avail: np.ndarray) -> int:
    valid = np.flatnonzero(avail)
    s = np.asarray(scores[valid], dtype=np.float64)
    local_order = np.lexsort((valid, -s))
    return int(valid[local_order[0]])


def maybe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def sample_users(users: List[int], max_users: Optional[int], seed: int) -> List[int]:
    users = list(users)
    if max_users is None or max_users >= len(users):
        return sorted(users)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(users), size=max_users, replace=False)
    return sorted([users[i] for i in idx])


def stable_json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def stable_hash(x: Any) -> str:
    s = stable_json_dumps(x)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fmt_mean_std(mean_val: float, std_val: float, digits: int = 5) -> str:
    if pd.isna(mean_val):
        return "-"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def fmt_percent(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{100.0 * x:.{digits}f}%"