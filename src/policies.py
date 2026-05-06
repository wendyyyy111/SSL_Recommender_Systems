
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Config, get_policy_eta
from .utils import topk_from_scores, stable_argmax_with_mask



def build_state_vectors(cfg: Config, genre2idx: Dict[str, int], n_genres: int):
    c0 = np.zeros(n_genres, dtype=np.float32)
    c1 = np.zeros(n_genres, dtype=np.float32)

    idx0 = [genre2idx[g] for g in cfg.bucket0 if g in genre2idx]
    idx1 = [genre2idx[g] for g in cfg.bucket1 if g in genre2idx]

    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Bucket genres not found in MovieLens genre vocabulary.")

    c0[idx0] = 1.0 / len(idx0)
    c1[idx1] = 1.0 / len(idx1)
    d = c1 - c0
    return c0, c1, d, idx0, idx1


def build_item_semantic_scores(item_genre_mat: np.ndarray, c0: np.ndarray, c1: np.ndarray, d: np.ndarray):
    bonus0 = item_genre_mat @ c0
    bonus1 = item_genre_mat @ c1
    item_d = item_genre_mat @ d
    return bonus0.astype(np.float32), bonus1.astype(np.float32), item_d.astype(np.float32)
def get_user_state_bonus_vectors(
    cfg: Config,
    u: int,
    bonus0_base: np.ndarray,
    bonus1_base: np.ndarray,
    state_meta: Dict[str, Any],
):
    family = state_meta["state_family"]

    if family in {"semantic_linear", "sparse_bucket"}:
        return bonus0_base, bonus1_base

    if family == "history_conditioned":
        item_side0 = state_meta["item_side0"]
        item_side1 = state_meta["item_side1"]
        user_pref0 = float(state_meta["user_pref0"][u])
        user_pref1 = float(state_meta["user_pref1"][u])
        w = float(state_meta["mix_w"])

        # user-conditioned item bonus
        b0 = (1.0 - w) * item_side0 + w * user_pref0 * item_side0
        b1 = (1.0 - w) * item_side1 + w * user_pref1 * item_side1
        return np.asarray(b0, dtype=np.float32), np.asarray(b1, dtype=np.float32)

    raise ValueError(f"Unknown state_family={family}")
def build_hidden_state_item_bonuses(
    cfg: Config,
    data_dict,
    audit_users: List[int],
    train_pos: Dict[int, List[int]],
):
    item_genre_mat = data_dict["item_genre_mat"]
    genre2idx = data_dict["genre2idx"]
    n_genres = data_dict["n_genres"]
    n_items = data_dict["n_items"]

    family = getattr(cfg, "state_family", "semantic_linear")

    if family == "semantic_linear":
        c0, c1, d, bucket0_idx, bucket1_idx = build_state_vectors(cfg, genre2idx, n_genres)
        bonus0_raw, bonus1_raw, item_d = build_item_semantic_scores(item_genre_mat, c0, c1, d)
        meta = {
            "state_family": family,
            "bucket0_idx": bucket0_idx,
            "bucket1_idx": bucket1_idx,
            "state_desc_0": cfg.state_desc_0,
            "state_desc_1": cfg.state_desc_1,
        }
        return bonus0_raw, bonus1_raw, item_d, meta

    elif family == "sparse_bucket":
        idx0 = [genre2idx[g] for g in cfg.sparse_bucket0 if g in genre2idx]
        idx1 = [genre2idx[g] for g in cfg.sparse_bucket1 if g in genre2idx]
        if len(idx0) == 0 or len(idx1) == 0:
            raise ValueError("Sparse bucket genres not found in genre vocabulary.")

        c0 = np.zeros(n_genres, dtype=np.float32)
        c1 = np.zeros(n_genres, dtype=np.float32)
        c0[idx0] = 1.0
        c1[idx1] = 1.0
        c0 /= float(np.sum(c0))
        c1 /= float(np.sum(c1))
        d = c1 - c0

        bonus0_raw, bonus1_raw, item_d = build_item_semantic_scores(item_genre_mat, c0, c1, d)
        meta = {
            "state_family": family,
            "bucket0_idx": idx0,
            "bucket1_idx": idx1,
            "state_desc_0": f"a short-term preference for sparse bucket {cfg.sparse_bucket0}",
            "state_desc_1": f"a short-term preference for sparse bucket {cfg.sparse_bucket1}",
        }
        return bonus0_raw, bonus1_raw, item_d, meta

    elif family == "history_conditioned":
        # item-side prototypes
        c0, c1, d, bucket0_idx, bucket1_idx = build_state_vectors(cfg, genre2idx, n_genres)
        item_side0 = item_genre_mat @ c0
        item_side1 = item_genre_mat @ c1

        # history-side user scalars
        h_user = compute_user_history_genre_dist(audit_users, train_pos, item_genre_mat)

        user_pref0 = {}
        user_pref1 = {}
        for u in audit_users:
            hu = h_user[u]
            user_pref0[u] = float(np.dot(hu, c0))
            user_pref1[u] = float(np.dot(hu, c1))

        # return item-side defaults + per-user modulation info in meta
        # item_d still kept for scalar RA penalty construction
        item_d = (item_side1 - item_side0).astype(np.float32)

        meta = {
            "state_family": family,
            "bucket0_idx": bucket0_idx,
            "bucket1_idx": bucket1_idx,
            "item_side0": np.asarray(item_side0, dtype=np.float32),
            "item_side1": np.asarray(item_side1, dtype=np.float32),
            "user_pref0": user_pref0,
            "user_pref1": user_pref1,
            "mix_w": float(cfg.history_state_mix_weight),
            "state_desc_0": "a short-term intent interacting with the user's historical romance/drama tendency",
            "state_desc_1": "a short-term intent interacting with the user's historical action/thriller tendency",
        }

        # place-holder raw bonuses; actual per-user bonus computed later
        bonus0_raw = np.asarray(item_side0, dtype=np.float32)
        bonus1_raw = np.asarray(item_side1, dtype=np.float32)
        return bonus0_raw, bonus1_raw, item_d, meta

    else:
        raise ValueError(f"Unknown state_family={family}")
def serve_topk_agnostic(base_scores_u: np.ndarray, train_items: List[int], topk: int):
    s = np.asarray(base_scores_u, dtype=np.float64).copy()
    if len(train_items) > 0:
        s[np.array(train_items, dtype=np.int64)] = -np.inf
    return topk_from_scores(s, topk)


def serve_topk_stateaware(
    base_scores_u: np.ndarray,
    train_items: List[int],
    topk: int,
    eta: float,
    item_bonus: np.ndarray,
    rerank_M: int = 100,
):
    scores = np.asarray(base_scores_u, dtype=np.float64).copy()
    bonus = np.asarray(item_bonus, dtype=np.float64)

    if len(train_items) > 0:
        scores[np.array(train_items, dtype=np.int64)] = -np.inf

    n_items = int(scores.shape[0])
    M = min(int(rerank_M), n_items)

    cand_idx = np.argpartition(-scores, M - 1)[:M]
    cand_scores = scores[cand_idx]
    rerank_scores = cand_scores + float(eta) * bonus[cand_idx]

    order = np.lexsort((cand_idx, -rerank_scores))
    top = cand_idx[order[:topk]]

    return np.asarray(top, dtype=np.int32)


def serve_topk_ra(
    base_scores_u: np.ndarray,
    train_items: List[int],
    topk: int,
    eta: float,
    item_bonus: np.ndarray,
    item_d: np.ndarray,
    z: int,
    lam: float,
    rerank_M: int = 100,
):
    sigma = 1.0 if z == 1 else -1.0

    scores = np.asarray(base_scores_u, dtype=np.float64).copy()
    bonus = np.asarray(item_bonus, dtype=np.float64)
    dvec = np.asarray(item_d, dtype=np.float64)

    if len(train_items) > 0:
        scores[np.array(train_items, dtype=np.int64)] = -np.inf

    n_items = int(scores.shape[0])
    M = min(int(rerank_M), n_items)

    cand_idx = np.argpartition(-scores, M - 1)[:M]
    cand_scores = scores[cand_idx]
    rerank_scores = cand_scores + float(eta) * bonus[cand_idx] - sigma * float(lam) * dvec[cand_idx]

    order = np.lexsort((cand_idx, -rerank_scores))
    top = cand_idx[order[:topk]]

    return np.asarray(top, dtype=np.int32)


def serve_topk_scalar_lambda(
    base_scores_u: np.ndarray,
    train_items: List[int],
    topk: int,
    eta: float,
    item_bonus: np.ndarray,
    item_d: np.ndarray,
    z: int,
    lam: float,
    rerank_M: int = 100,
    normalize_scores: bool = True,
    eps: float = 1e-8,
):
    sigma = 1.0 if z == 1 else -1.0

    scores = np.asarray(base_scores_u, dtype=np.float64).copy()
    bonus = np.asarray(item_bonus, dtype=np.float64)
    dvec = np.asarray(item_d, dtype=np.float64)

    if len(train_items) > 0:
        scores[np.array(train_items, dtype=np.int64)] = -np.inf

    n_items = int(scores.shape[0])
    M = min(int(rerank_M), n_items)

    cand_idx = np.argpartition(-scores, M - 1)[:M]
    cand_scores = scores[cand_idx].copy()
    cand_bonus = bonus[cand_idx].copy()
    cand_dvec = dvec[cand_idx].copy()

    if normalize_scores:
        s_mu = float(np.mean(cand_scores[np.isfinite(cand_scores)]))
        s_sd = float(np.std(cand_scores[np.isfinite(cand_scores)]))
        if s_sd > eps:
            cand_scores = (cand_scores - s_mu) / s_sd

        b_mu = float(np.mean(cand_bonus))
        b_sd = float(np.std(cand_bonus))
        if b_sd > eps:
            cand_bonus = (cand_bonus - b_mu) / b_sd

        d_mu = float(np.mean(cand_dvec))
        d_sd = float(np.std(cand_dvec))
        if d_sd > eps:
            cand_dvec = (cand_dvec - d_mu) / d_sd

    rerank_scores = cand_scores + float(eta) * cand_bonus - sigma * float(lam) * cand_dvec
    order = np.lexsort((cand_idx, -rerank_scores))
    top = cand_idx[order[:topk]]

    return np.asarray(top, dtype=np.int32)

def serve_topk_fullgamma(
    base_scores_u: np.ndarray,
    train_items: List[int],
    topk: int,
    eta: float,
    item_bonus: np.ndarray,
    item_feat_mat: np.ndarray,
    z: int,
    gamma: np.ndarray,
    rank_disc_coeffs: np.ndarray,
    rank_unw_coeff: float,
):
    n_items, n_feats = item_feat_mat.shape
    gamma = np.asarray(gamma, dtype=np.float64)

    if gamma.shape[0] != 2 * n_feats:
        raise ValueError(
            f"gamma dim mismatch: got {gamma.shape[0]}, expected {2 * n_feats}"
        )

    gamma_disc = gamma[:n_feats]
    gamma_unw = gamma[n_feats:]

    item_feat64 = np.asarray(item_feat_mat, dtype=np.float64)
    proj_disc = item_feat64 @ gamma_disc
    proj_unw = item_feat64 @ gamma_unw

    sigma = 1.0 if z == 1 else -1.0

    avail = np.ones(n_items, dtype=bool)
    if len(train_items) > 0:
        avail[np.array(train_items, dtype=np.int64)] = False

    base_plus_bonus = (
        np.asarray(base_scores_u, dtype=np.float64).copy()
        + float(eta) * np.asarray(item_bonus, dtype=np.float64)
    )

    out = np.empty(topk, dtype=np.int32)

    for r in range(topk):
        penalty_r = float(rank_disc_coeffs[r]) * proj_disc + float(rank_unw_coeff) * proj_unw
        s = base_plus_bonus - sigma * penalty_r
        i = stable_argmax_with_mask(s, avail)
        out[r] = i
        avail[i] = False

    return out
def sample_mix_transcripts(
    users: List[int],
    topk_agn: Dict[int, np.ndarray],
    topk_ra_z0: Dict[int, np.ndarray],
    topk_ra_z1: Dict[int, np.ndarray],
    alpha: float,
    seed: int,
    shared_coin: bool = True,
):
    topk_z0 = {}
    topk_z1 = {}
    rng = np.random.RandomState(seed)

    for u in users:
        if shared_coin:
            use_agn = rng.rand() < alpha
            topk_z0[u] = topk_agn[u] if use_agn else topk_ra_z0[u]
            topk_z1[u] = topk_agn[u] if use_agn else topk_ra_z1[u]
        else:
            use_agn_0 = rng.rand() < alpha
            use_agn_1 = rng.rand() < alpha
            topk_z0[u] = topk_agn[u] if use_agn_0 else topk_ra_z0[u]
            topk_z1[u] = topk_agn[u] if use_agn_1 else topk_ra_z1[u]

    return topk_z0, topk_z1


def min_alpha_to_certify(ra_val_qleak: float, rad_val: float, tau: float) -> float:
    """
    Need: (1 - alpha) * ra_val_qleak + rad_val <= tau
    """
    if rad_val > tau + 1e-15:
        return float("inf")
    if ra_val_qleak <= 1e-15:
        return 0.0
    slack = tau - rad_val
    alpha_need = 1.0 - (slack / ra_val_qleak)
    return float(np.clip(alpha_need, 0.0, 1.0))


def make_mix_alpha_candidates(
    cfg: Config,
    alpha_needs: Optional[List[float]] = None,
) -> List[float]:
    return sorted(set(float(a) for a in cfg.alpha_grid))
