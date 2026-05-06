import math
from typing import Dict, List, Optional

import numpy as np

from .utils import ndcg_at_k, recall_at_k
def build_rank_weights(k: int):
    w = np.array([1.0 / math.log2(r + 2.0) for r in range(k)], dtype=np.float32)
    return w, float(w.sum())


def transcript_stats(topk_items: np.ndarray, item_genre_mat: np.ndarray, rank_w: np.ndarray, rank_w_sum: float,
                     bucket0_idx: List[int], bucket1_idx: List[int]):
    mat = item_genre_mat[topk_items]
    E = (rank_w[:, None] * mat).sum(axis=0) / rank_w_sum
    Ebar = mat.mean(axis=0)

    denom = float(E.sum()) + 1e-12
    Etilde = E / denom
    entropy = float(-(Etilde * np.log(Etilde + 1e-12)).sum() / math.log(len(Etilde)))
    concentration = float(Etilde.max())

    item_bucket0 = (mat[:, bucket0_idx].sum(axis=1) > 0).astype(np.float32)
    item_bucket1 = (mat[:, bucket1_idx].sum(axis=1) > 0).astype(np.float32)

    bucket0_mass = float((rank_w * item_bucket0).sum() / rank_w_sum)
    bucket1_mass = float((rank_w * item_bucket1).sum() / rank_w_sum)

    def _warn_if_out_of_range(name: str, x):
        arr = np.asarray(x, dtype=np.float32)
        xmin = float(np.min(arr))
        xmax = float(np.max(arr))
        if xmin < -1e-6 or xmax > 1.0 + 1e-6:
            print(f"[WARN] transcript_stats::{name} out of range: min={xmin:.6f}, max={xmax:.6f}")

    _warn_if_out_of_range("E", E)
    _warn_if_out_of_range("Ebar", Ebar)
    _warn_if_out_of_range("entropy", [entropy])
    _warn_if_out_of_range("concentration", [concentration])
    _warn_if_out_of_range("bucket0_mass", [bucket0_mass])
    _warn_if_out_of_range("bucket1_mass", [bucket1_mass])

    return {
        "E": E.astype(np.float32),
        "Ebar": Ebar.astype(np.float32),
        "entropy": entropy,
        "concentration": concentration,
        "bucket0_mass": bucket0_mass,
        "bucket1_mass": bucket1_mass,
    }

def build_query_vector(
    topk_items: np.ndarray,
    user_hist_genre: np.ndarray,
    item_genre_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
):
    s = transcript_stats(topk_items, item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx)

    E = np.clip(np.asarray(s["E"], dtype=np.float32), 0.0, 1.0)
    Ebar = np.clip(np.asarray(s["Ebar"], dtype=np.float32), 0.0, 1.0)
    g_cent = np.clip((1.0 + E - np.asarray(user_hist_genre, dtype=np.float32)) / 2.0, 0.0, 1.0)
    entropy = np.array([np.clip(float(s["entropy"]), 0.0, 1.0)], dtype=np.float32)
    concentration = np.array([np.clip(float(s["concentration"]), 0.0, 1.0)], dtype=np.float32)

    thresholds = [0.1, 0.2, 0.3, 0.4]
    thr_feats = []
    for a in thresholds:
        thr_feats.append(1.0 if float(s["bucket0_mass"]) >= a else 0.0)
        thr_feats.append(1.0 if float(s["bucket1_mass"]) >= a else 0.0)
    thr_feats = np.array(thr_feats, dtype=np.float32)

    q = np.concatenate(
        [E, Ebar, g_cent, entropy, concentration, thr_feats],
        axis=0
    ).astype(np.float32)
    return q

def build_query_vector_diag(
    topk_items: np.ndarray,
    user_hist_genre: np.ndarray,
    item_genre_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
):
    s = transcript_stats(topk_items, item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx)

    E = np.clip(np.asarray(s["E"], dtype=np.float32), 0.0, 1.0)
    Ebar = np.clip(np.asarray(s["Ebar"], dtype=np.float32), 0.0, 1.0)
    g_cent = np.clip((1.0 + E - np.asarray(user_hist_genre, dtype=np.float32)) / 2.0, 0.0, 1.0)

    entropy = np.array([np.clip(float(s["entropy"]), 0.0, 1.0)], dtype=np.float32)
    concentration = np.array([np.clip(float(s["concentration"]), 0.0, 1.0)], dtype=np.float32)

    raw_gap = float(s["bucket1_mass"]) - float(s["bucket0_mass"])
    bucket_gap = np.array([0.5 * (1.0 + np.clip(raw_gap, -1.0, 1.0))], dtype=np.float32)

    q = np.concatenate(
        [E, Ebar, g_cent, entropy, concentration, bucket_gap],
        axis=0
    ).astype(np.float32)
    return q


def build_query_names_main_paper(
    genre_list: List[str],
    include_thresholds: bool = True,
):
    names = []
    for g in genre_list:
        names.append(f"E_disc::{g}")
    for g in genre_list:
        names.append(f"E_unw::{g}")
    for g in genre_list:
        names.append(f"E_cent::{g}")
    names.append("entropy")
    names.append("concentration")
    if include_thresholds:
        for a in [0.1, 0.2, 0.3, 0.4]:
            names.append(f"bucket0_ge_{a}")
            names.append(f"bucket1_ge_{a}")
    return names

def build_query_names_diag(
    genre_list: List[str],
):
    names = []
    for g in genre_list:
        names.append(f"E_disc::{g}")
    for g in genre_list:
        names.append(f"E_unw::{g}")
    for g in genre_list:
        names.append(f"E_cent::{g}")
    names.append("entropy")
    names.append("concentration")
    names.append("bucket_gap")
    return names



def build_cert_query_names(base_feat_names: List[str]):
    names = []
    for n in base_feat_names:
        names.append(f"CERT_E_disc::{n}")
    for n in base_feat_names:
        names.append(f"CERT_E_unw::{n}")
    return names


def build_cert_query_vector(
    topk_items: np.ndarray,
    item_feat_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
):
    mat = item_feat_mat[topk_items]
    E_disc = (rank_w[:, None] * mat).sum(axis=0) / rank_w_sum
    E_unw = mat.mean(axis=0)
    return np.concatenate([E_disc, E_unw], axis=0).astype(np.float32)


def evaluate_cert_policy_metrics(
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    heldout: Dict[int, int],
    item_feat_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    heldout_z1: Optional[Dict[int, int]] = None,
    debug_query_ranges_flag: bool = False,
    debug_max_users_print: int = 0,
):
    ndcgs = []
    recalls = []
    q_diffs = []

    for i, u in enumerate(users):
        t0 = topk_z0[u]
        t1 = topk_z1[u]

        target0 = heldout[u]
        target1 = heldout[u] if heldout_z1 is None else heldout_z1[u]

        nd0 = ndcg_at_k(t0, target0)
        nd1 = ndcg_at_k(t1, target1)
        rc0 = recall_at_k(t0, target0)
        rc1 = recall_at_k(t1, target1)

        ndcgs.append(0.5 * (nd0 + nd1))
        recalls.append(0.5 * (rc0 + rc1))

        q0 = build_cert_query_vector(t0, item_feat_mat, rank_w, rank_w_sum)
        q1 = build_cert_query_vector(t1, item_feat_mat, rank_w, rank_w_sum)

        if debug_query_ranges_flag and i < int(debug_max_users_print):
            debug_check_query_ranges(f"add user={u} q0", q0)
            debug_check_query_ranges(f"add user={u} q1", q1)

        q_diffs.append(q1 - q0)

    q_diffs = np.stack(q_diffs, axis=0)
    mean_q_diff = q_diffs.mean(axis=0)
    qleak = float(np.max(np.abs(mean_q_diff)))

    return {
        "ndcg": float(np.mean(ndcgs)),
        "recall": float(np.mean(recalls)),
        "qleak": qleak,
        "mean_q_diff": mean_q_diff.astype(np.float32),
    }

def ucb_radius(num_queries: int, num_candidates_total: int, delta: float, num_users: int):
    return math.sqrt(2.0 * math.log(2.0 * num_queries * num_candidates_total / delta) / num_users)

def top_leakage_rows(policy: str, split: str, mean_q_diff: np.ndarray, query_names: List[str], topn: int = 5):
    idx = np.argsort(np.abs(mean_q_diff))[::-1][:topn]
    rows = []
    for rank, j in enumerate(idx, start=1):
        rows.append(
            {
                "policy": policy,
                "split": split,
                "rank": rank,
                "query": query_names[j],
                "signed_diff": float(mean_q_diff[j]),
                "abs_diff": float(abs(mean_q_diff[j])),
            }
        )
    return rows


def top_leakage_summary(mean_q_diff: np.ndarray, query_names: List[str]) -> Dict[str, Any]:
    j = int(np.argmax(np.abs(mean_q_diff)))
    return {
        "top_query": query_names[j],
        "top_signed_diff": float(mean_q_diff[j]),
        "top_abs_diff": float(abs(mean_q_diff[j])),
    }