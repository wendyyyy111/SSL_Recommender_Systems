import json
import os
import time
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .attackers import run_statistical_attacker_suite
from .config import Config, get_policy_eta
from .data import (
    build_state_conditioned_audit_data,
    compute_user_history_genre_dist,
    split_dev_val_test_users,
)
from .llm_audit import run_llm_audit_suite
from .models import train_and_precompute_ranker_scores
from .policies import (
    build_hidden_state_item_bonuses,
    get_user_state_bonus_vectors,
    make_mix_alpha_candidates,
    min_alpha_to_certify,
    sample_mix_transcripts,
    serve_topk_agnostic,
    serve_topk_scalar_lambda,
    serve_topk_stateaware,
)
from .queries import (
    build_cert_query_names,
    build_query_names_diag,
    build_query_names_main_paper,
    build_rank_weights,
    evaluate_cert_policy_metrics,
    evaluate_policy_metrics,
    evaluate_policy_metrics_diag,
    top_leakage_rows,
    top_leakage_summary,
    ucb_radius,
)
from .reporting import (
    aggregate_auxiliary_outputs,
    aggregate_results,
    build_selection_summary,
    prettify_aggregate,
)
from .utils import ensure_dir, maybe_float, normalize_bonus, seed_everything
def build_dev_leak_aligned_item_risk(
    aware_dev_mean_q_diff: np.ndarray,
    item_feat_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    center: bool = True,
    scale_by_std: bool = True,
    clip_z: float = 3.0,
    eps: float = 1e-8,
):
    """
    Map additive cert leakage direction -> per-item scalar risk.

    aware_dev_mean_q_diff shape = [2F]
      first F  : CERT_E_disc::*
      second F : CERT_E_unw::*

    We collapse the additive query leakage direction into one scalar per item.
    """
    q = np.asarray(aware_dev_mean_q_diff, dtype=np.float32)
    Fdim = item_feat_mat.shape[1]
    assert q.shape[0] == 2 * Fdim, f"Expected 2F dims, got {q.shape[0]} vs 2*{Fdim}"

    w_disc = q[:Fdim]
    w_unw = q[Fdim:]

    # rank-aware collapse
    avg_disc_coeff = float(np.mean(rank_w / rank_w_sum))
    avg_unw_coeff = float(1.0 / len(rank_w))

    w = avg_disc_coeff * w_disc + avg_unw_coeff * w_unw
    risk = np.asarray(item_feat_mat, dtype=np.float32) @ np.asarray(w, dtype=np.float32)

    if center:
        risk = risk - float(np.mean(risk))

    if scale_by_std:
        s = float(np.std(risk))
        if s > eps:
            risk = risk / s
        risk = np.clip(risk, -clip_z, clip_z)

    return risk.astype(np.float32)



def build_exposure_risk_feature(
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    n_items: int,
    rho: float = 1.0,
    clip_B: float = 4.0,
    rank_slice: Optional[Tuple[int, int]] = None,
):
    if n_items <= 0:
        raise ValueError("n_items must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive for smoothing.")

    c0 = np.zeros(n_items, dtype=np.float64)
    c1 = np.zeros(n_items, dtype=np.float64)

    lo, hi = 0, None
    if rank_slice is not None:
        lo = int(rank_slice[0])
        hi = int(rank_slice[1])

    for u in users:
        z0 = np.asarray(topk_z0[u], dtype=np.int64)
        z1 = np.asarray(topk_z1[u], dtype=np.int64)

        z0 = z0[lo:hi]
        z1 = z1[lo:hi]

        np.add.at(c0, z0, 1.0)
        np.add.at(c1, z1, 1.0)

    p0 = (c0 + rho) / (float(c0.sum()) + rho * n_items)
    p1 = (c1 + rho) / (float(c1.sum()) + rho * n_items)

    risk = np.abs(np.log(p1 / p0))
    B = max(float(clip_B), 1e-8)
    risk = np.minimum(risk / B, 1.0).astype(np.float32)

    if not np.all(np.isfinite(risk)):
        raise ValueError("Exposure risk feature contains non-finite values.")
    return risk

def build_context_stratified_exposure_risk_features(
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    n_items: int,
    rho: float,
    clip_B: float,
    rank_strata: Tuple[Tuple[int, int], ...],
):
    feat_cols = []
    feat_names = []

    global_risk = build_exposure_risk_feature(
        users=users,
        topk_z0=topk_z0,
        topk_z1=topk_z1,
        n_items=n_items,
        rho=rho,
        clip_B=clip_B,
        rank_slice=None,
    )
    feat_cols.append(global_risk[:, None])
    feat_names.append("exposure_risk_all")

    for lo, hi in rank_strata:
        risk = build_exposure_risk_feature(
            users=users,
            topk_z0=topk_z0,
            topk_z1=topk_z1,
            n_items=n_items,
            rho=rho,
            clip_B=clip_B,
            rank_slice=(int(lo), int(hi)),
        )
        feat_cols.append(risk[:, None])
        feat_names.append(f"exposure_risk_rank_{int(lo)}_{int(hi)}")

    return np.concatenate(feat_cols, axis=1).astype(np.float32), feat_names

def build_additive_feature_bank(
    data_dict,
    cfg: Config,
    dev_users: List[int],
    topk_dev_z0: Dict[int, np.ndarray],
    topk_dev_z1: Dict[int, np.ndarray],
):
    item_genre_mat = data_dict["item_genre_mat"]
    n_items, _ = item_genre_mat.shape

    feat_list = []
    feat_names = []

    # 1) genre one-hot
    feat_list.append(item_genre_mat.astype(np.float32))
    feat_names.extend([f"genre::{g}" for g in data_dict["genre_list"]])

    # 2) bucket indicators
    genre2idx = data_dict["genre2idx"]
    b0 = np.zeros(n_items, dtype=np.float32)
    b1 = np.zeros(n_items, dtype=np.float32)

    for g in cfg.bucket0:
        if g in genre2idx:
            b0 = np.maximum(b0, item_genre_mat[:, genre2idx[g]].astype(np.float32))

    for g in cfg.bucket1:
        if g in genre2idx:
            b1 = np.maximum(b1, item_genre_mat[:, genre2idx[g]].astype(np.float32))

    feat_list.append(b0[:, None])
    feat_list.append(b1[:, None])
    feat_names.extend(["bucket0_indicator", "bucket1_indicator"])

    # 3) exposure risk estimated on DEV transcripts only
    if cfg.use_exposure_risk_feature:
        if getattr(cfg, "use_context_stratified_risk", False):
            risk_mat, risk_names = build_context_stratified_exposure_risk_features(
                users=dev_users,
                topk_z0=topk_dev_z0,
                topk_z1=topk_dev_z1,
                n_items=n_items,
                rho=cfg.risk_smoothing,
                clip_B=cfg.risk_clip_B,
                rank_strata=getattr(cfg, "risk_rank_strata", ((0, cfg.topk),)),
            )
            feat_list.append(risk_mat)
            feat_names.extend(risk_names)
        else:
            risk = build_exposure_risk_feature(
                users=dev_users,
                topk_z0=topk_dev_z0,
                topk_z1=topk_dev_z1,
                n_items=n_items,
                rho=cfg.risk_smoothing,
                clip_B=cfg.risk_clip_B,
            )
            feat_list.append(risk[:, None])
            feat_names.append("exposure_risk")

    item_feat_mat = np.concatenate(feat_list, axis=1).astype(np.float32)

    if item_feat_mat.shape[0] != n_items:
        raise AssertionError("Feature bank row count mismatch with items.")
    if item_feat_mat.shape[1] != len(feat_names):
        raise AssertionError("Feature bank name count mismatch with feature dimension.")
    if not np.all(np.isfinite(item_feat_mat)):
        raise AssertionError("Feature bank contains non-finite values.")

    return item_feat_mat, feat_names


def select_best_candidate_from_df(
    df: pd.DataFrame,
    family: str,
    tau: float,
):
    """
    family: 'RA' or 'RA+Mix'
    Return:
      best_row_dict, is_feasible
    """
    if family not in {"RA", "RA+Mix"}:
        raise ValueError(f"Unknown family: {family}")

    feasible = df[df["val_ucb"] <= tau].copy()

    if len(feasible) == 0:
        fallback = df.sort_values(
            ["val_ucb", "val_qleak", "val_ndcg"],
            ascending=[True, True, False]
        ).iloc[0].to_dict()
        return fallback, False

    if family == "RA":
        feasible = feasible.sort_values(
            ["val_ndcg", "val_recall", "gamma_l1", "val_qleak"],
            ascending=[False, False, True, True]
        )
    else:
        feasible = feasible.sort_values(
            ["val_ndcg", "val_recall", "alpha", "val_qleak"],
            ascending=[False, False, True, True]
        )

    return feasible.iloc[0].to_dict(), True


def summarize_best_by_tau(
    df: pd.DataFrame,
    family: str,
    tau_list: List[float],
) -> pd.DataFrame:
    rows = []
    for tau in tau_list:
        best_row, feasible = select_best_candidate_from_df(df, family, float(tau))
        row = dict(best_row)
        row["tau"] = float(tau)
        row["feasible"] = int(feasible)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_over_tau(
    tau_df: pd.DataFrame,
    family: str,
) -> Dict[str, Any]:
    feasible = tau_df[tau_df["feasible"] == 1].copy()

    if len(feasible) == 0:
        return tau_df.sort_values(
            ["val_ucb", "val_qleak", "val_ndcg"],
            ascending=[True, True, False]
        ).iloc[0].to_dict()

    if family == "RA":
        feasible = feasible.sort_values(
            ["val_ndcg", "val_recall", "tau", "gamma_l1", "val_qleak"],
            ascending=[False, False, True, True, True]
        )
    elif family == "RA+Mix":
        feasible = feasible.sort_values(
            ["val_ndcg", "val_recall", "tau", "alpha", "val_qleak"],
            ascending=[False, False, True, True, True]
        )
    else:
        raise ValueError(f"Unknown family={family}")

    return feasible.iloc[0].to_dict()
def get_tau_candidates(cfg: Config) -> List[float]:
    if getattr(cfg, "tau_selection_mode", "fixed") == "best_certified":
        return sorted(set(float(t) for t in cfg.tau_grid))
    return [float(cfg.tau)]

def build_scalar_lambda_mix_rows(
    cfg: Config,
    scalar_lambda_candidates: List[Dict[str, Any]],
    tau_candidates: List[float],
    agn_dev_G: Dict[str, Any],
    agn_val_G: Dict[str, Any],
    agn_test_G: Dict[str, Any],
    agn_dev_add: Dict[str, Any],
    agn_val_add: Dict[str, Any],
    agn_test_add: Dict[str, Any],
    full_query_names: List[str],
    cert_query_names: List[str],
    rad_val_main: float,
    rad_val_cert: float,
):
    alpha_candidates = make_mix_alpha_candidates(cfg)
    rows = []

    for cand in scalar_lambda_candidates:
        lam = float(cand["lambda"])

        ra_dev_add = cand["dev_metrics"]
        ra_val_add = cand["val_metrics"]
        ra_test_add = cand["test_metrics"]

        ra_dev_main = cand["dev_metrics_G"]
        ra_val_main = cand["val_metrics_G"]
        ra_test_main = cand["test_metrics_G"]

        alpha_need = min_alpha_to_certify(float(ra_val_main["qleak"]), rad_val_main, cfg.tau)

        for alpha in alpha_candidates:
            dev_ndcg = (1.0 - alpha) * float(ra_dev_main["ndcg"]) + alpha * float(agn_dev_G["ndcg"])
            dev_recall = (1.0 - alpha) * float(ra_dev_main["recall"]) + alpha * float(agn_dev_G["recall"])
            dev_mean_q_diff = (1.0 - alpha) * ra_dev_main["mean_q_diff"] + alpha * agn_dev_G["mean_q_diff"]
            dev_qleak = float(np.max(np.abs(dev_mean_q_diff)))

            val_ndcg = (1.0 - alpha) * float(ra_val_main["ndcg"]) + alpha * float(agn_val_G["ndcg"])
            val_recall = (1.0 - alpha) * float(ra_val_main["recall"]) + alpha * float(agn_val_G["recall"])
            val_mean_q_diff = (1.0 - alpha) * ra_val_main["mean_q_diff"] + alpha * agn_val_G["mean_q_diff"]
            val_qleak = float(np.max(np.abs(val_mean_q_diff)))
            val_diag = top_leakage_summary(val_mean_q_diff, full_query_names)

            test_ndcg = (1.0 - alpha) * float(ra_test_main["ndcg"]) + alpha * float(agn_test_G["ndcg"])
            test_recall = (1.0 - alpha) * float(ra_test_main["recall"]) + alpha * float(agn_test_G["recall"])
            test_mean_q_diff = (1.0 - alpha) * ra_test_main["mean_q_diff"] + alpha * agn_test_G["mean_q_diff"]
            test_qleak = float(np.max(np.abs(test_mean_q_diff)))
            test_diag = top_leakage_summary(test_mean_q_diff, full_query_names)

            dev_ndcg_add = (1.0 - alpha) * float(ra_dev_add["ndcg"]) + alpha * float(agn_dev_add["ndcg"])
            dev_recall_add = (1.0 - alpha) * float(ra_dev_add["recall"]) + alpha * float(agn_dev_add["recall"])
            dev_mean_q_diff_add = (1.0 - alpha) * ra_dev_add["mean_q_diff"] + alpha * agn_dev_add["mean_q_diff"]
            dev_qleak_add = float(np.max(np.abs(dev_mean_q_diff_add)))

            val_ndcg_add = (1.0 - alpha) * float(ra_val_add["ndcg"]) + alpha * float(agn_val_add["ndcg"])
            val_recall_add = (1.0 - alpha) * float(ra_val_add["recall"]) + alpha * float(agn_val_add["recall"])
            val_mean_q_diff_add = (1.0 - alpha) * ra_val_add["mean_q_diff"] + alpha * agn_val_add["mean_q_diff"]
            val_qleak_add = float(np.max(np.abs(val_mean_q_diff_add)))
            val_diag_add = top_leakage_summary(val_mean_q_diff_add, cert_query_names)

            test_ndcg_add = (1.0 - alpha) * float(ra_test_add["ndcg"]) + alpha * float(agn_test_add["ndcg"])
            test_recall_add = (1.0 - alpha) * float(ra_test_add["recall"]) + alpha * float(agn_test_add["recall"])
            test_mean_q_diff_add = (1.0 - alpha) * ra_test_add["mean_q_diff"] + alpha * agn_test_add["mean_q_diff"]
            test_qleak_add = float(np.max(np.abs(test_mean_q_diff_add)))
            test_diag_add = top_leakage_summary(test_mean_q_diff_add, cert_query_names)

            rows.append(
                {
                    "family": "RA+Mix",
                    "lambda": lam,
                    "candidate_key": f"lambda_{lam:.6f}",
                    "gamma_iter": None,
                    "gamma_variant": "scalar_lambda",
                    "gamma_l1": abs(lam),
                    "gamma_l2": abs(lam),
                    "alpha": alpha,
                    "alpha_needed": None if not np.isfinite(alpha_need) else alpha_need,

                    "dev_ndcg": dev_ndcg,
                    "dev_recall": dev_recall,
                    "dev_qleak": dev_qleak,
                    "val_ndcg": val_ndcg,
                    "val_recall": val_recall,
                    "val_qleak": val_qleak,
                    "val_ucb": val_qleak + rad_val_main,
                    "val_top_query": val_diag["top_query"],
                    "val_top_abs_diff": val_diag["top_abs_diff"],
                    "test_ndcg": test_ndcg,
                    "test_recall": test_recall,
                    "test_qleak": test_qleak,
                    "test_top_query": test_diag["top_query"],
                    "test_top_abs_diff": test_diag["top_abs_diff"],
                    "certified": int((val_qleak + rad_val_main) <= cfg.tau + 1e-12),

                    "dev_ndcg_add": dev_ndcg_add,
                    "dev_recall_add": dev_recall_add,
                    "dev_qleak_add": dev_qleak_add,
                    "val_ndcg_add": val_ndcg_add,
                    "val_recall_add": val_recall_add,
                    "val_qleak_add": val_qleak_add,
                    "val_ucb_add": val_qleak_add + rad_val_cert,
                    "val_top_query_add": val_diag_add["top_query"],
                    "val_top_abs_diff_add": val_diag_add["top_abs_diff"],
                    "test_ndcg_add": test_ndcg_add,
                    "test_recall_add": test_recall_add,
                    "test_qleak_add": test_qleak_add,
                    "test_top_query_add": test_diag_add["top_query"],
                    "test_top_abs_diff_add": test_diag_add["top_abs_diff"],
                    "certified_add": int((val_qleak_add + rad_val_cert) <= cfg.tau),
                }
            )

    return rows

def build_scalar_lambda_ra_rows(
    cfg: Config,
    scalar_lambda_candidates: List[Dict[str, Any]],
    full_query_names: List[str],
    cert_query_names: List[str],
    rad_val_main: float,
    rad_val_cert: float,
):
    rows = []

    for cand in scalar_lambda_candidates:
        lam = float(cand["lambda"])

        dev_add = cand["dev_metrics"]
        val_add = cand["val_metrics"]
        test_add = cand["test_metrics"]

        dev_main = cand["dev_metrics_G"]
        val_main = cand["val_metrics_G"]
        test_main = cand["test_metrics_G"]

        val_diag_main = top_leakage_summary(val_main["mean_q_diff"], full_query_names)
        test_diag_main = top_leakage_summary(test_main["mean_q_diff"], full_query_names)
        val_diag_add = top_leakage_summary(val_add["mean_q_diff"], cert_query_names)
        test_diag_add = top_leakage_summary(test_add["mean_q_diff"], cert_query_names)

        rows.append(
            {
                "family": "RA",
                "lambda": lam,
                "candidate_key": f"lambda_{lam:.6f}",
                "gamma_iter": None,
                "gamma_variant": "scalar_lambda",
                "gamma_l1": abs(lam),
                "gamma_l2": abs(lam),
                "alpha": None,

                "dev_ndcg": dev_main["ndcg"],
                "dev_recall": dev_main["recall"],
                "dev_qleak": dev_main["qleak"],
                "val_ndcg": val_main["ndcg"],
                "val_recall": val_main["recall"],
                "val_qleak": val_main["qleak"],
                "val_ucb": val_main["qleak"] + rad_val_main,
                "val_top_query": val_diag_main["top_query"],
                "val_top_abs_diff": val_diag_main["top_abs_diff"],
                "test_ndcg": test_main["ndcg"],
                "test_recall": test_main["recall"],
                "test_qleak": test_main["qleak"],
                "test_top_query": test_diag_main["top_query"],
                "test_top_abs_diff": test_diag_main["top_abs_diff"],
                "certified": int((val_main["qleak"] + rad_val_main) <= cfg.tau),

                "dev_ndcg_add": dev_add["ndcg"],
                "dev_recall_add": dev_add["recall"],
                "dev_qleak_add": dev_add["qleak"],
                "val_ndcg_add": val_add["ndcg"],
                "val_recall_add": val_add["recall"],
                "val_qleak_add": val_add["qleak"],
                "val_ucb_add": val_add["qleak"] + rad_val_cert,
                "val_top_query_add": val_diag_add["top_query"],
                "val_top_abs_diff_add": val_diag_add["top_abs_diff"],
                "test_ndcg_add": test_add["ndcg"],
                "test_recall_add": test_add["recall"],
                "test_qleak_add": test_add["qleak"],
                "test_top_query_add": test_diag_add["top_query"],
                "test_top_abs_diff_add": test_diag_add["top_abs_diff"],
                "certified_add": int((val_add["qleak"] + rad_val_cert) <= cfg.tau),
            }
        )

    return rows
def build_scalar_lambda_mix_rows(
    cfg: Config,
    scalar_lambda_candidates: List[Dict[str, Any]],
    tau_candidates: List[float],
    agn_dev_G: Dict[str, Any],
    agn_val_G: Dict[str, Any],
    agn_test_G: Dict[str, Any],
    agn_dev_add: Dict[str, Any],
    agn_val_add: Dict[str, Any],
    agn_test_add: Dict[str, Any],
    full_query_names: List[str],
    cert_query_names: List[str],
    rad_val_main: float,
    rad_val_cert: float,
):
    alpha_candidates = make_mix_alpha_candidates(cfg)
    rows = []

    for cand in scalar_lambda_candidates:
        lam = float(cand["lambda"])

        ra_dev_add = cand["dev_metrics"]
        ra_val_add = cand["val_metrics"]
        ra_test_add = cand["test_metrics"]

        ra_dev_main = cand["dev_metrics_G"]
        ra_val_main = cand["val_metrics_G"]
        ra_test_main = cand["test_metrics_G"]

        alpha_need = min_alpha_to_certify(float(ra_val_main["qleak"]), rad_val_main, cfg.tau)

        for alpha in alpha_candidates:
            dev_ndcg = (1.0 - alpha) * float(ra_dev_main["ndcg"]) + alpha * float(agn_dev_G["ndcg"])
            dev_recall = (1.0 - alpha) * float(ra_dev_main["recall"]) + alpha * float(agn_dev_G["recall"])
            dev_mean_q_diff = (1.0 - alpha) * ra_dev_main["mean_q_diff"] + alpha * agn_dev_G["mean_q_diff"]
            dev_qleak = float(np.max(np.abs(dev_mean_q_diff)))

            val_ndcg = (1.0 - alpha) * float(ra_val_main["ndcg"]) + alpha * float(agn_val_G["ndcg"])
            val_recall = (1.0 - alpha) * float(ra_val_main["recall"]) + alpha * float(agn_val_G["recall"])
            val_mean_q_diff = (1.0 - alpha) * ra_val_main["mean_q_diff"] + alpha * agn_val_G["mean_q_diff"]
            val_qleak = float(np.max(np.abs(val_mean_q_diff)))
            val_diag = top_leakage_summary(val_mean_q_diff, full_query_names)

            test_ndcg = (1.0 - alpha) * float(ra_test_main["ndcg"]) + alpha * float(agn_test_G["ndcg"])
            test_recall = (1.0 - alpha) * float(ra_test_main["recall"]) + alpha * float(agn_test_G["recall"])
            test_mean_q_diff = (1.0 - alpha) * ra_test_main["mean_q_diff"] + alpha * agn_test_G["mean_q_diff"]
            test_qleak = float(np.max(np.abs(test_mean_q_diff)))
            test_diag = top_leakage_summary(test_mean_q_diff, full_query_names)

            dev_ndcg_add = (1.0 - alpha) * float(ra_dev_add["ndcg"]) + alpha * float(agn_dev_add["ndcg"])
            dev_recall_add = (1.0 - alpha) * float(ra_dev_add["recall"]) + alpha * float(agn_dev_add["recall"])
            dev_mean_q_diff_add = (1.0 - alpha) * ra_dev_add["mean_q_diff"] + alpha * agn_dev_add["mean_q_diff"]
            dev_qleak_add = float(np.max(np.abs(dev_mean_q_diff_add)))

            val_ndcg_add = (1.0 - alpha) * float(ra_val_add["ndcg"]) + alpha * float(agn_val_add["ndcg"])
            val_recall_add = (1.0 - alpha) * float(ra_val_add["recall"]) + alpha * float(agn_val_add["recall"])
            val_mean_q_diff_add = (1.0 - alpha) * ra_val_add["mean_q_diff"] + alpha * agn_val_add["mean_q_diff"]
            val_qleak_add = float(np.max(np.abs(val_mean_q_diff_add)))
            val_diag_add = top_leakage_summary(val_mean_q_diff_add, cert_query_names)

            test_ndcg_add = (1.0 - alpha) * float(ra_test_add["ndcg"]) + alpha * float(agn_test_add["ndcg"])
            test_recall_add = (1.0 - alpha) * float(ra_test_add["recall"]) + alpha * float(agn_test_add["recall"])
            test_mean_q_diff_add = (1.0 - alpha) * ra_test_add["mean_q_diff"] + alpha * agn_test_add["mean_q_diff"]
            test_qleak_add = float(np.max(np.abs(test_mean_q_diff_add)))
            test_diag_add = top_leakage_summary(test_mean_q_diff_add, cert_query_names)

            rows.append(
                {
                    "family": "RA+Mix",
                    "lambda": lam,
                    "candidate_key": f"lambda_{lam:.6f}",
                    "gamma_iter": None,
                    "gamma_variant": "scalar_lambda",
                    "gamma_l1": abs(lam),
                    "gamma_l2": abs(lam),
                    "alpha": alpha,
                    "alpha_needed": None if not np.isfinite(alpha_need) else alpha_need,

                    "dev_ndcg": dev_ndcg,
                    "dev_recall": dev_recall,
                    "dev_qleak": dev_qleak,
                    "val_ndcg": val_ndcg,
                    "val_recall": val_recall,
                    "val_qleak": val_qleak,
                    "val_ucb": val_qleak + rad_val_main,
                    "val_top_query": val_diag["top_query"],
                    "val_top_abs_diff": val_diag["top_abs_diff"],
                    "test_ndcg": test_ndcg,
                    "test_recall": test_recall,
                    "test_qleak": test_qleak,
                    "test_top_query": test_diag["top_query"],
                    "test_top_abs_diff": test_diag["top_abs_diff"],
                    "certified": int((val_qleak + rad_val_main) <= cfg.tau + 1e-12),

                    "dev_ndcg_add": dev_ndcg_add,
                    "dev_recall_add": dev_recall_add,
                    "dev_qleak_add": dev_qleak_add,
                    "val_ndcg_add": val_ndcg_add,
                    "val_recall_add": val_recall_add,
                    "val_qleak_add": val_qleak_add,
                    "val_ucb_add": val_qleak_add + rad_val_cert,
                    "val_top_query_add": val_diag_add["top_query"],
                    "val_top_abs_diff_add": val_diag_add["top_abs_diff"],
                    "test_ndcg_add": test_ndcg_add,
                    "test_recall_add": test_recall_add,
                    "test_qleak_add": test_qleak_add,
                    "test_top_query_add": test_diag_add["top_query"],
                    "test_top_abs_diff_add": test_diag_add["top_abs_diff"],
                    "certified_add": int((val_qleak_add + rad_val_cert) <= cfg.tau),
                }
            )

    return rows


def build_scalar_lambda_sweep_rows(
    scalar_lambda_candidates: List[Dict[str, Any]],
    full_query_names: List[str],
    cert_query_names: List[str],
    rad_val_main: float,
    rad_val_cert: float,
    tau: float,
):
    rows = []

    for cand in scalar_lambda_candidates:
        lam = float(cand["lambda"])

        val_G = cand["val_metrics_G"]
        test_G = cand["test_metrics_G"]
        val_add = cand["val_metrics"]
        test_add = cand["test_metrics"]

        val_top_G = top_leakage_summary(val_G["mean_q_diff"], full_query_names)
        test_top_G = top_leakage_summary(test_G["mean_q_diff"], full_query_names)
        val_top_add = top_leakage_summary(val_add["mean_q_diff"], cert_query_names)
        test_top_add = top_leakage_summary(test_add["mean_q_diff"], cert_query_names)

        rows.append(
            {
                "lambda": lam,

                # main = full G
                "val_ndcg": float(val_G["ndcg"]),
                "val_recall": float(val_G["recall"]),
                "val_qleak": float(val_G["qleak"]),
                "val_ucb": float(val_G["qleak"] + rad_val_main),
                "test_ndcg": float(test_G["ndcg"]),
                "test_recall": float(test_G["recall"]),
                "test_qleak": float(test_G["qleak"]),
                "certified": int((val_G["qleak"] + rad_val_main) <= tau),
                "val_top_query": val_top_G["top_query"],
                "val_top_abs_diff": float(val_top_G["top_abs_diff"]),
                "test_top_query": test_top_G["top_query"],
                "test_top_abs_diff": float(test_top_G["top_abs_diff"]),

                # keep additive diagnostics
                "val_ndcg_add": float(val_add["ndcg"]),
                "val_recall_add": float(val_add["recall"]),
                "val_qleak_add": float(val_add["qleak"]),
                "val_ucb_add": float(val_add["qleak"] + rad_val_cert),
                "test_ndcg_add": float(test_add["ndcg"]),
                "test_recall_add": float(test_add["recall"]),
                "test_qleak_add": float(test_add["qleak"]),
                "val_top_query_add": val_top_add["top_query"],
                "val_top_abs_diff_add": float(val_top_add["top_abs_diff"]),
                "test_top_query_add": test_top_add["top_query"],
                "test_top_abs_diff_add": float(test_top_add["top_abs_diff"]),
            }
        )

    return rows

def build_scalar_candidate_debug_rows(
    scalar_lambda_candidates: List[Dict[str, Any]],
    aware_topk_z0: Dict[int, np.ndarray],
    aware_topk_z1: Dict[int, np.ndarray],
    val_users: List[int],
):
    rows = []
    for cand in scalar_lambda_candidates:
        lam = float(cand["lambda"])

        overlaps0 = []
        overlaps1 = []
        exact0 = []
        exact1 = []

        for u in val_users:
            t0 = cand["topk_z0"][u]
            t1 = cand["topk_z1"][u]
            a0 = aware_topk_z0[u]
            a1 = aware_topk_z1[u]

            overlaps0.append(topk_overlap(t0, a0))
            overlaps1.append(topk_overlap(t1, a1))
            exact0.append(float(np.array_equal(t0, a0)))
            exact1.append(float(np.array_equal(t1, a1)))

        rows.append({
            "lambda": lam,
            "val_ndcg_main": float(cand["val_metrics_G"]["ndcg"]),
            "val_qleak_main": float(cand["val_metrics_G"]["qleak"]),
            "test_qleak_main": float(cand["test_metrics_G"]["qleak"]),
            "val_ndcg_add": float(cand["val_metrics"]["ndcg"]),
            "val_qleak_add": float(cand["val_metrics"]["qleak"]),
            "mean_overlap_z0_vs_aware": float(np.mean(overlaps0)),
            "mean_overlap_z1_vs_aware": float(np.mean(overlaps1)),
            "exact_match_z0_vs_aware": float(np.mean(exact0)),
            "exact_match_z1_vs_aware": float(np.mean(exact1)),
        })
    return pd.DataFrame(rows)


def report_heldout_alignment(users: List[int], heldout: Dict[int, int], bonus0: np.ndarray, bonus1: np.ndarray):
    idx = np.array([heldout[u] for u in users], dtype=np.int64)
    b0 = bonus0[idx]
    b1 = bonus1[idx]

    rows = {
        "heldout_mean_bonus0": float(np.mean(b0)),
        "heldout_mean_bonus1": float(np.mean(b1)),
        "heldout_frac_bucket0": float(np.mean(b0 > 0)),
        "heldout_frac_bucket1": float(np.mean(b1 > 0)),
        "heldout_frac_neither": float(np.mean((b0 == 0) & (b1 == 0))),
        "heldout_frac_both": float(np.mean((b0 > 0) & (b1 > 0))),
    }
    print("[DEBUG] heldout alignment:", rows)
    return rows
def measure_policy_level_runtime_rows(
    cfg: Config,
    seed: int,
    audit_users: List[int],
    train_pos: Dict[int, List[int]],
    base_scores: Dict[int, np.ndarray],
    bonus0: np.ndarray,
    bonus1: np.ndarray,
    item_feat_mat: np.ndarray,
    item_d: np.ndarray,
    selected_ra_lambda: Optional[float],
    selected_mix_lambda: Optional[float],
    selected_ra_gamma: Optional[np.ndarray],
    selected_mix_gamma: Optional[np.ndarray],
    selected_mix_alpha: float,
    rank_w: np.ndarray,
    rank_w_sum: float,
):
    rows = []

    mode = getattr(cfg, "figure3_mode", "scalar_lambda")
    if mode not in {"scalar_lambda", "full_gamma"}:
        raise ValueError(f"Unknown figure3_mode={mode}")

    rank_disc_coeffs = (rank_w / rank_w_sum).astype(np.float32)
    rank_unw_coeff = float(1.0 / cfg.topk)

    def _measure(policy_name: str):
        total_sec_list = []
        num_calls = 0

        for rep in range(max(1, int(getattr(cfg, "policy_runtime_repeats", 1)))):
            rng = np.random.RandomState(seed * 1000 + 37 * (rep + 1))
            t0 = time.time()

            local_calls = 0
            for u in audit_users:
                base_u = base_scores[u]
                train_u = train_pos[u]

                if policy_name == "State-aware":
                    _ = serve_topk_stateaware(
                        base_u, train_u, cfg.topk, cfg.state_eta, bonus0, cfg.stateaware_rerank_M
                    )
                    _ = serve_topk_stateaware(
                        base_u, train_u, cfg.topk, cfg.state_eta, bonus1, cfg.stateaware_rerank_M
                    )
                    local_calls += 2

                elif policy_name == "State-independent":
                    _ = serve_topk_agnostic(base_u, train_u, cfg.topk)
                    _ = serve_topk_agnostic(base_u, train_u, cfg.topk)
                    local_calls += 2

                elif policy_name == "RA":
                    if mode == "scalar_lambda":
                        _ = serve_topk_scalar_lambda(
                            base_scores_u=base_u,
                            train_items=train_u,
                            topk=cfg.topk,
                            eta=get_policy_eta(cfg),
                            item_bonus=bonus0,
                            item_d=item_d,
                            z=0,
                            lam=float(selected_ra_lambda),
                            rerank_M=cfg.stateaware_rerank_M,
                        )
                        _ = serve_topk_scalar_lambda(
                            base_scores_u=base_u,
                            train_items=train_u,
                            topk=cfg.topk,
                            eta=get_policy_eta(cfg),
                            item_bonus=bonus1,
                            item_d=item_d,
                            z=1,
                            lam=float(selected_ra_lambda),
                            rerank_M=cfg.stateaware_rerank_M,
                        )
                    else:
                        if selected_ra_gamma is None:
                            raise ValueError("selected_ra_gamma is required for full_gamma runtime measurement.")
                        _ = serve_topk_fullgamma(
                            base_scores_u=base_u,
                            train_items=train_u,
                            topk=cfg.topk,
                            eta=get_policy_eta(cfg),
                            item_bonus=bonus0,
                            item_feat_mat=item_feat_mat,
                            z=0,
                            gamma=selected_ra_gamma,
                            rank_disc_coeffs=rank_disc_coeffs,
                            rank_unw_coeff=rank_unw_coeff,
                        )
                        _ = serve_topk_fullgamma(
                            base_scores_u=base_u,
                            train_items=train_u,
                            topk=cfg.topk,
                            eta=get_policy_eta(cfg),
                            item_bonus=bonus1,
                            item_feat_mat=item_feat_mat,
                            z=1,
                            gamma=selected_ra_gamma,
                            rank_disc_coeffs=rank_disc_coeffs,
                            rank_unw_coeff=rank_unw_coeff,
                        )
                    local_calls += 2

                elif policy_name == "RA+Mix":
                    for z, item_bonus in [(0, bonus0), (1, bonus1)]:
                        use_agn = rng.rand() < float(selected_mix_alpha)
                        if use_agn:
                            _ = serve_topk_agnostic(base_u, train_u, cfg.topk)
                        else:
                            if mode == "scalar_lambda":
                                _ = serve_topk_scalar_lambda(
                                    base_scores_u=base_u,
                                    train_items=train_u,
                                    topk=cfg.topk,
                                    eta=get_policy_eta(cfg),
                                    item_bonus=item_bonus,
                                    item_d=item_d,
                                    z=z,
                                    lam=float(selected_mix_lambda),
                                    rerank_M=cfg.stateaware_rerank_M,
                                )
                            else:
                                if selected_mix_gamma is None:
                                    raise ValueError("selected_mix_gamma is required for full_gamma runtime measurement.")
                                _ = serve_topk_fullgamma(
                                    base_scores_u=base_u,
                                    train_items=train_u,
                                    topk=cfg.topk,
                                    eta=get_policy_eta(cfg),
                                    item_bonus=item_bonus,
                                    item_feat_mat=item_feat_mat,
                                    z=z,
                                    gamma=selected_mix_gamma,
                                    rank_disc_coeffs=rank_disc_coeffs,
                                    rank_unw_coeff=rank_unw_coeff,
                                )
                        local_calls += 1
                else:
                    raise ValueError(policy_name)

            dt = float(time.time() - t0)
            total_sec_list.append(dt)
            num_calls = local_calls

        mean_total_sec = float(np.mean(total_sec_list))
        mean_ms_per_query = 1000.0 * mean_total_sec / max(1, num_calls)

        rows.append(
            {
                "seed": seed,
                "runtime_type": "policy_level",
                "policy": policy_name,
                "stage": None,
                "seconds": mean_total_sec,
                "num_queries": int(num_calls),
                "mean_ms_per_query": float(mean_ms_per_query),
            }
        )

    _measure("State-aware")
    _measure("State-independent")
    _measure("RA")
    _measure("RA+Mix")

    return rows


def export_tau_calibration_data(tau_sweep_root: str):
    rows = []

    if not os.path.exists(tau_sweep_root):
        return

    for subdir in sorted(os.listdir(tau_sweep_root)):
        if not subdir.startswith("tau_"):
            continue

        try:
            tau = float(subdir.split("_", 1)[1])
        except Exception:
            continue

        path = os.path.join(tau_sweep_root, subdir, "table1_all_seeds_full.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        for policy in ["RA", "RA+Mix"]:
            sub = df[df["policy"] == policy].copy()
            if len(sub) == 0:
                continue

            rows.append(
                {
                    "tau": tau,
                    "policy": policy,
                    "val_ucb_mean": float(sub["Val.UCB"].mean()),
                    "val_ucb_std": float(sub["Val.UCB"].std()),
                    "test_qleak_mean": float(sub["Test.qleak"].mean()),
                    "test_qleak_std": float(sub["Test.qleak"].std()),
                    "certified_rate": float(sub["Certified"].mean()),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(tau_sweep_root, "tau_calibration_summary.csv"), index=False)


def run_tau_sweep(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "tau_sweep")
    ensure_dir(sweep_root)

    summary_rows = []

    for tau in base_cfg.tau_grid:
        cfg_tau = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, f"tau_{tau:.3f}"),
            tau=float(tau),
            seeds=base_cfg.sweep_seeds,
            llm_enabled=base_cfg.sweep_llm_enabled,
        )
        out = run_experiment_bundle(cfg_tau, data_dict, bundle_name=f"tau={tau:.3f}")
        agg_df = out["agg_df"].copy()
        agg_df["tau"] = tau
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        sweep_df = pd.concat(summary_rows, axis=0, ignore_index=True)
        sweep_df.to_csv(os.path.join(sweep_root, "tau_sweep_summary.csv"), index=False)

    export_tau_calibration_data(sweep_root)
def run_one_seed(
    cfg: Config,
    data_dict,
    seed: int,
    ranker_name: Optional[str] = None,
    precomputed_base_scores: Optional[Dict[int, np.ndarray]] = None,
):
    seed_everything(seed)
    seed_dir = os.path.join(cfg.out_dir, f"seed_{seed}")
    ensure_dir(seed_dir)

    runtime_rows = []

    def _record_runtime(stage: str, t_start: float):
        runtime_rows.append(
            {
                "seed": seed,
                "runtime_type": "stage_level",
                "policy": None,
                "stage": stage,
                "seconds": float(time.time() - t_start),
                "num_queries": None,
                "mean_ms_per_query": None,
            }
        )

    raw_eligible_users = data_dict["eligible_users"]
    raw_train_pos = data_dict["train_pos"]
    raw_heldout = data_dict["heldout"]
    item_genre_mat = data_dict["item_genre_mat"]
    genre2idx = data_dict["genre2idx"]

    active_ranker = (ranker_name or getattr(cfg, "ranker_name", "bprmf")).lower()

    bonus0_raw, bonus1_raw, item_d, state_meta = build_hidden_state_item_bonuses(
        cfg=cfg,
        data_dict=data_dict,
        audit_users=raw_eligible_users,
        train_pos=raw_train_pos,
    )

    bucket0_idx = state_meta["bucket0_idx"]
    bucket1_idx = state_meta["bucket1_idx"]

    bonus0 = normalize_bonus(
        bonus0_raw,
        center=cfg.bonus_center,
        scale_by_std=cfg.bonus_scale_by_std,
        clip_abs=cfg.bonus_clip_abs,
        clip_z=cfg.bonus_clip_z,
        eps=cfg.bonus_eps,
    )
    bonus1 = normalize_bonus(
        bonus1_raw,
        center=cfg.bonus_center,
        scale_by_std=cfg.bonus_scale_by_std,
        clip_abs=cfg.bonus_clip_abs,
        clip_z=cfg.bonus_clip_z,
        eps=cfg.bonus_eps,
    )

    if cfg.use_paired_state_targets:
        eligible_users, train_pos, heldout_z0, heldout_z1 = build_state_conditioned_audit_data(
            cfg=cfg,
            data_dict=data_dict,
            bucket0_idx=bucket0_idx,
            bucket1_idx=bucket1_idx,
        )
        heldout = heldout_z0
    else:
        eligible_users = raw_eligible_users
        train_pos = raw_train_pos
        heldout = raw_heldout
        heldout_z1 = None

    dev_users, val_users, test_users = split_dev_val_test_users(
        eligible_users,
        seed,
        cfg.dev_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
    )
    audit_users = sorted(dev_users + val_users + test_users)

    if sorted(audit_users) != sorted(eligible_users):
        raise AssertionError("dev/val/test split does not partition eligible_users exactly.")
    if len(set(dev_users) & set(val_users)) > 0 or len(set(dev_users) & set(test_users)) > 0 or len(set(val_users) & set(test_users)) > 0:
        raise AssertionError("dev/val/test split overlap detected.")

    h_user = compute_user_history_genre_dist(audit_users, train_pos, item_genre_mat)

    train_data_dict = dict(data_dict)
    train_data_dict["eligible_users"] = audit_users
    train_data_dict["train_pos"] = train_pos
    train_data_dict["heldout"] = {u: heldout[u] for u in audit_users}
    train_data_dict["forbid_items"] = {u: {heldout[u]} for u in audit_users}

    t0 = time.time()
    if precomputed_base_scores is None:
        _, base_scores = train_and_precompute_ranker_scores(
            cfg=cfg,
            data_dict=train_data_dict,
            seed=seed,
            users=audit_users,
            ranker_name=active_ranker,
        )
    else:
        base_scores = precomputed_base_scores
    _record_runtime(f"train_{active_ranker}_and_precompute_scores", t0)

    t0 = time.time()
    topk_agn = {}
    topk_aware_z0 = {}
    topk_aware_z1 = {}

    for local_user_idx, u in enumerate(audit_users):
        base_u = base_scores[u]
        train_u = train_pos[u]

        bonus0_u, bonus1_u = get_user_state_bonus_vectors(
            cfg=cfg,
            u=u,
            bonus0_base=bonus0,
            bonus1_base=bonus1,
            state_meta=state_meta,
        )

        if cfg.debug_bonus_stats and local_user_idx < cfg.debug_max_users_print:
            debug_bonus_stats(f"user={u} bonus0_raw", bonus0_raw)
            debug_bonus_stats(f"user={u} bonus1_raw", bonus1_raw)
            debug_bonus_stats(f"user={u} bonus0_norm", bonus0_u)
            debug_bonus_stats(f"user={u} bonus1_norm", bonus1_u)

        topk_agn[u] = serve_topk_agnostic(base_u, train_u, cfg.topk)
        topk_aware_z0[u] = serve_topk_stateaware(
            base_scores_u=base_u,
            train_items=train_u,
            topk=cfg.topk,
            eta=cfg.state_eta,
            item_bonus=bonus0_u,
            rerank_M=cfg.stateaware_rerank_M,
        )
        topk_aware_z1[u] = serve_topk_stateaware(
            base_scores_u=base_u,
            train_items=train_u,
            topk=cfg.topk,
            eta=cfg.state_eta,
            item_bonus=bonus1_u,
            rerank_M=cfg.stateaware_rerank_M,
        )

    topk_aware_dev_z0 = {u: topk_aware_z0[u] for u in dev_users}
    topk_aware_dev_z1 = {u: topk_aware_z1[u] for u in dev_users}

    item_feat_mat, cert_base_feat_names = build_additive_feature_bank(
        data_dict=data_dict,
        cfg=cfg,
        dev_users=dev_users,
        topk_dev_z0=topk_aware_dev_z0,
        topk_dev_z1=topk_aware_dev_z1,
    )

    with open(os.path.join(seed_dir, "cert_feature_bank.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "num_base_features": int(item_feat_mat.shape[1]),
                "base_feature_names": cert_base_feat_names,
                "use_exposure_risk_feature": bool(cfg.use_exposure_risk_feature),
                "use_context_stratified_risk": bool(getattr(cfg, "use_context_stratified_risk", False)),
                "risk_rank_strata": list(getattr(cfg, "risk_rank_strata", tuple())),
                "risk_smoothing": float(cfg.risk_smoothing),
                "risk_clip_B": float(cfg.risk_clip_B),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _record_runtime("serve_baselines_and_build_feature_bank", t0)

    assert_topk_dict_valid("agnostic", audit_users, topk_agn, train_pos, cfg.topk)
    assert_topk_dict_valid("aware_z0", audit_users, topk_aware_z0, train_pos, cfg.topk)
    assert_topk_dict_valid("aware_z1", audit_users, topk_aware_z1, train_pos, cfg.topk)
    if cfg.debug_topk_overlap:
        debug_report_topk_overlap(
            name="state-aware z0 vs z1 topk overlap",
            tops0=topk_aware_z0,
            tops1=topk_aware_z1,
        )

    rank_w, rank_w_sum = build_rank_weights(cfg.topk)
    cert_query_names = build_cert_query_names(cert_base_feat_names)

    full_query_names = build_query_names_main_paper(data_dict["genre_list"], include_thresholds=True)
    diag_query_names = build_query_names_diag(data_dict["genre_list"])

    num_cert_queries = len(cert_query_names)
    num_full_queries = len(full_query_names)

    tau_candidates = get_tau_candidates(cfg)
    alpha_candidates = make_mix_alpha_candidates(cfg)

    num_ra_candidates = len(cfg.lambda_grid)
    num_mix_candidates = len(cfg.lambda_grid) * len(alpha_candidates)
    num_candidates_total = 2 + num_ra_candidates + num_mix_candidates

    rad_val_cert = ucb_radius(num_cert_queries, num_candidates_total, cfg.delta, len(val_users))
    rad_val_main = ucb_radius(num_full_queries, num_candidates_total, cfg.delta, len(val_users))

    print(f"[seed={seed}][ranker={active_ranker}] num_dev_users={len(dev_users)}, num_val_users={len(val_users)}, num_test_users={len(test_users)}")
    print(f"[seed={seed}] num_cert_queries={num_cert_queries}, num_full_queries={num_full_queries}, num_candidates_total={num_candidates_total}")
    print(f"[seed={seed}] num_ra_candidates={num_ra_candidates}, num_mix_candidates={num_mix_candidates}, num_alpha_candidates={len(alpha_candidates)}")
    print(f"[seed={seed}] rad_val_cert={rad_val_cert:.6f}, rad_val_main={rad_val_main:.6f}, tau={cfg.tau:.6f}")

    agn_dev = evaluate_cert_policy_metrics(
        dev_users, topk_agn, topk_agn, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    agn_val = evaluate_cert_policy_metrics(
        val_users, topk_agn, topk_agn, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    agn_test = evaluate_cert_policy_metrics(
        test_users, topk_agn, topk_agn, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )

    aware_dev = evaluate_cert_policy_metrics(
        dev_users, topk_aware_z0, topk_aware_z1, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    aware_val = evaluate_cert_policy_metrics(
        val_users, topk_aware_z0, topk_aware_z1, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    aware_test = evaluate_cert_policy_metrics(
        test_users, topk_aware_z0, topk_aware_z1, heldout,
        item_feat_mat, rank_w, rank_w_sum, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    if cfg.scalar_risk_mode == "dev_leak_linear":
        scalar_item_risk = build_dev_leak_aligned_item_risk(
            aware_dev_mean_q_diff=aware_dev["mean_q_diff"],
            item_feat_mat=item_feat_mat,
            rank_w=rank_w,
            rank_w_sum=rank_w_sum,
            center=cfg.scalar_risk_center,
            scale_by_std=cfg.scalar_risk_scale_by_std,
            clip_z=cfg.scalar_risk_clip_z,
            eps=cfg.bonus_eps,
        )
    else:
        scalar_item_risk = item_d.astype(np.float32)

    if cfg.debug_bonus_stats:
        debug_bonus_stats("scalar_item_risk", scalar_item_risk)

    agn_dev_G = evaluate_policy_metrics(
        dev_users, topk_agn, topk_agn, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    agn_val_G = evaluate_policy_metrics(
        val_users, topk_agn, topk_agn, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    agn_test_G = evaluate_policy_metrics(
        test_users, topk_agn, topk_agn, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )

    aware_dev_G = evaluate_policy_metrics(
        dev_users, topk_aware_z0, topk_aware_z1, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    aware_val_G = evaluate_policy_metrics(
        val_users, topk_aware_z0, topk_aware_z1, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    aware_test_G = evaluate_policy_metrics(
        test_users, topk_aware_z0, topk_aware_z1, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )

    aware_val_diag = evaluate_policy_metrics_diag(
        val_users, topk_aware_z0, topk_aware_z1, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )
    aware_test_diag = evaluate_policy_metrics_diag(
        test_users, topk_aware_z0, topk_aware_z1, heldout, h_user,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx, heldout_z1=heldout_z1,debug_query_ranges_flag=cfg.debug_query_ranges,
        debug_max_users_print=cfg.debug_max_users_print,
    )

    print(
        f"[seed={seed}] agnostic main(full-G) Val.qleak={agn_val_G['qleak']:.6f}, "
        f"Val.UCB={agn_val_G['qleak'] + rad_val_main:.6f}; "
        f"additive Val.qleak={agn_val['qleak']:.6f}, Val.UCB.add={agn_val['qleak'] + rad_val_cert:.6f}"
    )
    print(
        f"[seed={seed}] state-aware main(full-G) Val.qleak={aware_val_G['qleak']:.6f}, "
        f"Val.UCB={aware_val_G['qleak'] + rad_val_main:.6f}; "
        f"diag Val.qleak={aware_val_diag['qleak']:.6f}; "
        f"additive Val.qleak={aware_val['qleak']:.6f}, Val.UCB.add={aware_val['qleak'] + rad_val_cert:.6f}"
    )
    if cfg.debug_stateaware:
        debug_report_top_leak_dims("agnostic-main-val", agn_val_G["mean_q_diff"], full_query_names, topn=10)
        debug_report_top_leak_dims("aware-main-val", aware_val_G["mean_q_diff"], full_query_names, topn=10)
        debug_report_top_leak_dims("aware-diag-val", aware_val_diag["mean_q_diff"], diag_query_names, topn=10)
        debug_report_top_leak_dims("aware-additive-val", aware_val["mean_q_diff"], cert_query_names, topn=10)

    t0 = time.time()
    scalar_lambda_candidates = build_scalar_lambda_candidates(
        cfg=cfg,
        audit_users=audit_users,
        dev_users=dev_users,
        val_users=val_users,
        test_users=test_users,
        train_pos=train_pos,
        heldout=heldout,
        base_scores=base_scores,
        bonus0=bonus0,
        bonus1=bonus1,
        state_meta=state_meta,
        scalar_item_risk=scalar_item_risk,
        item_feat_mat=item_feat_mat,
        h_user=h_user,
        item_genre_mat=item_genre_mat,
        rank_w=rank_w,
        rank_w_sum=rank_w_sum,
        bucket0_idx=bucket0_idx,
        bucket1_idx=bucket1_idx,
        heldout_z1=heldout_z1,
    )
    _record_runtime("build_scalar_lambda_candidates", t0)
    cand_debug_df = build_scalar_candidate_debug_rows(
        scalar_lambda_candidates=scalar_lambda_candidates,
        aware_topk_z0=topk_aware_z0,
        aware_topk_z1=topk_aware_z1,
        val_users=val_users,
    )
    cand_debug_df.to_csv(os.path.join(seed_dir, "scalar_lambda_candidate_debug.csv"), index=False)

    gamma_rows = build_scalar_lambda_ra_rows(
        cfg=cfg,
        scalar_lambda_candidates=scalar_lambda_candidates,
        full_query_names=full_query_names,
        cert_query_names=cert_query_names,
        rad_val_main=rad_val_main,
        rad_val_cert=rad_val_cert,
    )
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(os.path.join(seed_dir, "fullgamma_candidates.csv"), index=False)

    ra_by_tau_df = summarize_best_by_tau(gamma_df, "RA", tau_candidates)
    ra_by_tau_df.to_csv(os.path.join(seed_dir, "ra_best_by_tau.csv"), index=False)

    heldout_alignment_raw = report_heldout_alignment(audit_users, heldout, bonus0_raw, bonus1_raw)
    heldout_alignment_norm = report_heldout_alignment(audit_users, heldout, bonus0, bonus1)

    heldout_alignment = {
        "raw": heldout_alignment_raw,
        "normalized": heldout_alignment_norm,
    }
    with open(os.path.join(seed_dir, "heldout_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(heldout_alignment, f, indent=2)

    if getattr(cfg, "tau_selection_mode", "fixed") == "best_certified":
        best_ra_tau_row = select_best_over_tau(ra_by_tau_df, "RA")
        selected_tau_ra = float(best_ra_tau_row["tau"])
        best_ra, ra_feasible = select_best_candidate_from_df(gamma_df, "RA", selected_tau_ra)
    else:
        selected_tau_ra = float(cfg.tau)
        best_ra, ra_feasible = select_best_candidate_from_df(gamma_df, "RA", selected_tau_ra)

    best_ra["feasible"] = ra_feasible

    lambda_to_cand = {float(c["lambda"]): c for c in scalar_lambda_candidates}
    best_ra_lam = float(best_ra["lambda"])
    best_ra_cand = lambda_to_cand[best_ra_lam]
    topk_ra_sel_z0 = best_ra_cand["topk_z0"]
    topk_ra_sel_z1 = best_ra_cand["topk_z1"]
    best_ra_iter = None
    best_ra_key = str(best_ra["candidate_key"])

    assert_topk_dict_valid("selected_ra_z0", audit_users, topk_ra_sel_z0, train_pos, cfg.topk)
    assert_topk_dict_valid("selected_ra_z1", audit_users, topk_ra_sel_z1, train_pos, cfg.topk)

    print(
        f"[seed={seed}] selected RA(main=scalar-lambda): key={best_ra_key}, "
        f"lambda={best_ra_lam:.6f}, selected_tau={selected_tau_ra:.6f}, "
        f"certified={bool(best_ra['feasible'])}, Val.UCB={float(best_ra['val_ucb']):.6f}"
    )

    t0 = time.time()
    mix_rows = build_scalar_lambda_mix_rows(
        cfg=cfg,
        scalar_lambda_candidates=scalar_lambda_candidates,
        tau_candidates=tau_candidates,
        agn_dev_G=agn_dev_G,
        agn_val_G=agn_val_G,
        agn_test_G=agn_test_G,
        agn_dev_add=agn_dev,
        agn_val_add=agn_val,
        agn_test_add=agn_test,
        full_query_names=full_query_names,
        cert_query_names=cert_query_names,
        rad_val_main=rad_val_main,
        rad_val_cert=rad_val_cert,
    )
    mix_df = pd.DataFrame(mix_rows)
    mix_df.to_csv(os.path.join(seed_dir, "fullgamma_mix_candidates.csv"), index=False)
    _record_runtime("build_mix_candidates", t0)

    mix_by_tau_df = summarize_best_by_tau(mix_df, "RA+Mix", tau_candidates)
    mix_by_tau_df.to_csv(os.path.join(seed_dir, "ramix_best_by_tau.csv"), index=False)

    if getattr(cfg, "tau_selection_mode", "fixed") == "best_certified":
        best_mix_tau_row = select_best_over_tau(mix_by_tau_df, "RA+Mix")
        selected_tau_mix = float(best_mix_tau_row["tau"])
        best_mix, mix_feasible = select_best_candidate_from_df(mix_df, "RA+Mix", selected_tau_mix)
    else:
        selected_tau_mix = float(cfg.tau)
        best_mix, mix_feasible = select_best_candidate_from_df(mix_df, "RA+Mix", selected_tau_mix)

    best_mix["feasible"] = mix_feasible

    best_mix_lam = float(best_mix["lambda"])
    best_mix_alpha = float(best_mix["alpha"])
    best_mix_cand = lambda_to_cand[best_mix_lam]
    best_mix_key = str(best_mix["candidate_key"])
    best_mix_iter = None

    print(
        f"[seed={seed}] selected RA+Mix(main=scalar-lambda): key={best_mix_key}, "
        f"lambda={best_mix_lam:.6f}, alpha={best_mix_alpha:.4f}, "
        f"selected_tau={selected_tau_mix:.6f}, "
        f"certified={bool(best_mix['feasible'])}, Val.UCB={float(best_mix['val_ucb']):.6f}"
    )

    best_over_tau_rows = [
        {
            "seed": seed,
            "policy": "RA",
            "selected_tau": float(selected_tau_ra),
            "feasible": int(ra_feasible),
            "candidate_key": str(best_ra["candidate_key"]),
            "gamma_iter": np.nan,
            "gamma_l1": float(best_ra["gamma_l1"]),
            "alpha": np.nan,
            "lambda": float(best_ra["lambda"]),
            "val_ndcg": float(best_ra["val_ndcg"]),
            "val_recall": float(best_ra["val_recall"]),
            "val_qleak": float(best_ra["val_qleak"]),
            "val_ucb": float(best_ra["val_ucb"]),
            "test_ndcg": float(best_ra["test_ndcg"]),
            "test_recall": float(best_ra["test_recall"]),
            "test_qleak": float(best_ra["test_qleak"]),
        },
        {
            "seed": seed,
            "policy": "RA+Mix",
            "selected_tau": float(selected_tau_mix),
            "feasible": int(mix_feasible),
            "candidate_key": str(best_mix["candidate_key"]),
            "gamma_iter": np.nan,
            "gamma_l1": float(best_mix["gamma_l1"]),
            "alpha": float(best_mix["alpha"]),
            "lambda": float(best_mix["lambda"]),
            "val_ndcg": float(best_mix["val_ndcg"]),
            "val_recall": float(best_mix["val_recall"]),
            "val_qleak": float(best_mix["val_qleak"]),
            "val_ucb": float(best_mix["val_ucb"]),
            "test_ndcg": float(best_mix["test_ndcg"]),
            "test_recall": float(best_mix["test_recall"]),
            "test_qleak": float(best_mix["test_qleak"]),
        },
    ]
    pd.DataFrame(best_over_tau_rows).to_csv(
        os.path.join(seed_dir, "best_over_tau_selection.csv"),
        index=False,
    )

    best_mix_val_mean_q_diff_main = (1.0 - best_mix_alpha) * best_mix_cand["val_metrics_G"]["mean_q_diff"] + best_mix_alpha * agn_val_G["mean_q_diff"]
    best_mix_test_mean_q_diff_main = (1.0 - best_mix_alpha) * best_mix_cand["test_metrics_G"]["mean_q_diff"] + best_mix_alpha * agn_test_G["mean_q_diff"]

    best_mix_val_mean_q_diff_add = (1.0 - best_mix_alpha) * best_mix_cand["val_metrics"]["mean_q_diff"] + best_mix_alpha * agn_val["mean_q_diff"]
    best_mix_test_mean_q_diff_add = (1.0 - best_mix_alpha) * best_mix_cand["test_metrics"]["mean_q_diff"] + best_mix_alpha * agn_test["mean_q_diff"]

    top_leak_rows_main = []
    top_leak_rows_main.extend(top_leakage_rows("State-aware", "val", aware_val_G["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("State-aware", "test", aware_test_G["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("State-independent", "val", agn_val_G["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("State-independent", "test", agn_test_G["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("RA", "val", best_ra_cand["val_metrics_G"]["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("RA", "test", best_ra_cand["test_metrics_G"]["mean_q_diff"], full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("RA+Mix", "val", best_mix_val_mean_q_diff_main, full_query_names, topn=5))
    top_leak_rows_main.extend(top_leakage_rows("RA+Mix", "test", best_mix_test_mean_q_diff_main, full_query_names, topn=5))
    pd.DataFrame(top_leak_rows_main).to_csv(os.path.join(seed_dir, "top_leaking_queries_fullG.csv"), index=False)

    top_leak_rows_diag = []
    top_leak_rows_diag.extend(top_leakage_rows("State-aware", "val", aware_val_diag["mean_q_diff"], diag_query_names, topn=5))
    top_leak_rows_diag.extend(top_leakage_rows("State-aware", "test", aware_test_diag["mean_q_diff"], diag_query_names, topn=5))
    pd.DataFrame(top_leak_rows_diag).to_csv(os.path.join(seed_dir, "top_leaking_queries_diag.csv"), index=False)

    top_leak_rows_add = []
    top_leak_rows_add.extend(top_leakage_rows("State-aware", "val", aware_val["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("State-aware", "test", aware_test["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("State-independent", "val", agn_val["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("State-independent", "test", agn_test["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("RA", "val", best_ra_cand["val_metrics"]["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("RA", "test", best_ra_cand["test_metrics"]["mean_q_diff"], cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("RA+Mix", "val", best_mix_val_mean_q_diff_add, cert_query_names, topn=5))
    top_leak_rows_add.extend(top_leakage_rows("RA+Mix", "test", best_mix_test_mean_q_diff_add, cert_query_names, topn=5))
    pd.DataFrame(top_leak_rows_add).to_csv(os.path.join(seed_dir, "top_leaking_cert_queries.csv"), index=False)

    ra_top_main = top_leakage_summary(best_ra_cand["val_metrics_G"]["mean_q_diff"], full_query_names)
    mix_top_main = top_leakage_summary(best_mix_val_mean_q_diff_main, full_query_names)

    if getattr(cfg, "figure3_mode", "scalar_lambda") == "scalar_lambda":
        t0 = time.time()
        scalar_rows = build_scalar_lambda_sweep_rows(
            scalar_lambda_candidates=scalar_lambda_candidates,
            full_query_names=full_query_names,
            cert_query_names=cert_query_names,
            rad_val_main=rad_val_main,
            rad_val_cert=rad_val_cert,
            tau=cfg.tau,
        )
        pd.DataFrame(scalar_rows).to_csv(
            os.path.join(seed_dir, "figure3_scalar_lambda_sweep.csv"), index=False
        )
        _record_runtime("build_scalar_lambda_sweep", t0)

    t0 = time.time()
    attacker_out = run_statistical_attacker_suite(
        cfg=cfg,
        seed=seed,
        seed_dir=seed_dir,
        val_users=val_users,
        test_users=test_users,
        topk_aware_z0=topk_aware_z0,
        topk_aware_z1=topk_aware_z1,
        topk_agn=topk_agn,
        topk_ra_sel_z0=topk_ra_sel_z0,
        topk_ra_sel_z1=topk_ra_sel_z1,
        best_ra_cand=best_ra_cand,
        best_mix_cand=best_mix_cand,
        best_mix_alpha=best_mix_alpha,
        item_genre_mat=item_genre_mat,
        rank_w=rank_w,
        rank_w_sum=rank_w_sum,
        bucket0_idx=bucket0_idx,
        bucket1_idx=bucket1_idx,
        h_user=h_user,
        heldout=heldout,
        item_feat_mat=item_feat_mat,
        agn_val_G=agn_val_G,
        agn_test_G=agn_test_G,
        heldout_z1=heldout_z1,
    )
    aware_auc = attacker_out["aware_auc"]
    agn_auc = attacker_out["agn_auc"]
    ra_auc = attacker_out["ra_auc"]
    ra_auc_cert_only = attacker_out["ra_auc_cert_only"]
    mix_auc = attacker_out["mix_auc"]
    mix_auc_cert = attacker_out["mix_auc_cert"]
    _record_runtime("eval_attackers", t0)

    llm_summary_by_policy = {
        p: {
            "LLM.AUC": np.nan,
            "LLM.SingleAcc": np.nan,
            "LLM.PairAcc": np.nan,
            "LLM.SingleN": 0,
            "LLM.PairN": 0,
            "Monitor.Score": np.nan,
            "StateReveal.Score": np.nan,
            "PrivacyInvasive.Score": np.nan,
        }
        for p in POLICY_ORDER
    }

    if cfg.llm_enabled:
        t0 = time.time()
        llm_summary_by_policy = run_llm_audit_suite(
            cfg=cfg,
            data_dict=data_dict,
            seed=seed,
            seed_dir=seed_dir,
            test_users=test_users,
            topk_aware_z0=topk_aware_z0,
            topk_aware_z1=topk_aware_z1,
            topk_agn=topk_agn,
            topk_ra_sel_z0=topk_ra_sel_z0,
            topk_ra_sel_z1=topk_ra_sel_z1,
            best_mix_cand=best_mix_cand,
            best_mix_alpha=best_mix_alpha,
        )
        _record_runtime("eval_llm", t0)

    selected_ra_gamma_runtime = None
    selected_mix_gamma_runtime = None

    policy_runtime_rows = measure_policy_level_runtime_rows(
        cfg=cfg,
        seed=seed,
        audit_users=audit_users,
        train_pos=train_pos,
        base_scores=base_scores,
        bonus0=bonus0,
        bonus1=bonus1,
        item_feat_mat=item_feat_mat,
        item_d=item_d,
        selected_ra_lambda=best_ra_lam,
        selected_mix_lambda=best_mix_lam,
        selected_ra_gamma=selected_ra_gamma_runtime,
        selected_mix_gamma=selected_mix_gamma_runtime,
        selected_mix_alpha=best_mix_alpha,
        rank_w=rank_w,
        rank_w_sum=rank_w_sum,
    )
    runtime_rows.extend(policy_runtime_rows)

    state_aware_cert = bool((aware_val_G["qleak"] + rad_val_main) <= cfg.tau)
    state_ind_cert = bool((agn_val_G["qleak"] + rad_val_main) <= cfg.tau)
    ra_cert = bool(float(best_ra["val_ucb"]) <= float(selected_tau_ra))
    mix_cert = bool(float(best_mix["val_ucb"]) <= float(selected_tau_mix))

    final_rows = [
        {
            "seed": seed,
            "ranker_name": active_ranker,
            "policy": "State-aware",
            "eval_k": int(cfg.topk),
            "Certified": int(state_aware_cert),
            "ndcg": float(aware_test_G["ndcg"]),
            "recall": float(aware_test_G["recall"]),
            "Val.UCB": float(aware_val_G["qleak"] + rad_val_main),
            "Test.qleak": float(aware_test_G["qleak"]),
            "Stat.AUC": maybe_float(aware_auc),
            "LLM.AUC": llm_summary_by_policy["State-aware"]["LLM.AUC"],
            "LLM.SingleAcc": llm_summary_by_policy["State-aware"]["LLM.SingleAcc"],
            "LLM.PairAcc": llm_summary_by_policy["State-aware"]["LLM.PairAcc"],
            "Monitor.Score": llm_summary_by_policy["State-aware"]["Monitor.Score"],
            "StateReveal.Score": llm_summary_by_policy["State-aware"]["StateReveal.Score"],
            "PrivacyInvasive.Score": llm_summary_by_policy["State-aware"]["PrivacyInvasive.Score"],
            "selected_tau": float(cfg.tau),
            "selected_gamma_iter": None,
            "selected_gamma_l1": None,
            "selected_alpha": None,
            "selected_lambda": None,
            "Val.UCB.add": float(aware_val["qleak"] + rad_val_cert),
            "Test.qleak.add": float(aware_test["qleak"]),
            "state_family": str(cfg.state_family),
            "state_eta": float(cfg.state_eta),
            "policy_eta": float(get_policy_eta(cfg)),
        },
        {
            "seed": seed,
            "ranker_name": active_ranker,
            "policy": "State-independent",
            "eval_k": int(cfg.topk),
            "Certified": int(state_ind_cert),
            "ndcg": float(agn_test_G["ndcg"]),
            "recall": float(agn_test_G["recall"]),
            "Val.UCB": float(agn_val_G["qleak"] + rad_val_main),
            "Test.qleak": float(agn_test_G["qleak"]),
            "Stat.AUC": maybe_float(agn_auc),
            "LLM.AUC": llm_summary_by_policy["State-independent"]["LLM.AUC"],
            "LLM.SingleAcc": llm_summary_by_policy["State-independent"]["LLM.SingleAcc"],
            "LLM.PairAcc": llm_summary_by_policy["State-independent"]["LLM.PairAcc"],
            "Monitor.Score": llm_summary_by_policy["State-independent"]["Monitor.Score"],
            "StateReveal.Score": llm_summary_by_policy["State-independent"]["StateReveal.Score"],
            "PrivacyInvasive.Score": llm_summary_by_policy["State-independent"]["PrivacyInvasive.Score"],
            "selected_tau": float(cfg.tau),
            "selected_gamma_iter": None,
            "selected_gamma_l1": None,
            "selected_alpha": None,
            "selected_lambda": None,
            "Val.UCB.add": float(agn_val["qleak"] + rad_val_cert),
            "Test.qleak.add": float(agn_test["qleak"]),
            "state_family": str(cfg.state_family),
            "state_eta": float(cfg.state_eta),
            "policy_eta": float(get_policy_eta(cfg)),
        },
        {
            "seed": seed,
            "ranker_name": active_ranker,
            "policy": "RA",
            "eval_k": int(cfg.topk),
            "Certified": int(ra_cert),
            "ndcg": float(best_ra["test_ndcg"]),
            "recall": float(best_ra["test_recall"]),
            "Val.UCB": float(best_ra["val_ucb"]),
            "Test.qleak": float(best_ra["test_qleak"]),
            "Stat.AUC": maybe_float(ra_auc),
            "LLM.AUC": llm_summary_by_policy["RA"]["LLM.AUC"],
            "LLM.SingleAcc": llm_summary_by_policy["RA"]["LLM.SingleAcc"],
            "LLM.PairAcc": llm_summary_by_policy["RA"]["LLM.PairAcc"],
            "Monitor.Score": llm_summary_by_policy["RA"]["Monitor.Score"],
            "StateReveal.Score": llm_summary_by_policy["RA"]["StateReveal.Score"],
            "PrivacyInvasive.Score": llm_summary_by_policy["RA"]["PrivacyInvasive.Score"],
            "selected_tau": float(selected_tau_ra),
            "selected_gamma_iter": None,
            "selected_gamma_l1": float(best_ra["gamma_l1"]),
            "selected_alpha": None,
            "selected_lambda": float(best_ra_lam),
            "Val.UCB.add": float(best_ra["val_ucb_add"]),
            "Test.qleak.add": float(best_ra["test_qleak_add"]),
            "state_family": str(cfg.state_family),
            "state_eta": float(cfg.state_eta),
            "policy_eta": float(get_policy_eta(cfg)),
        },
        {
            "seed": seed,
            "ranker_name": active_ranker,
            "policy": "RA+Mix",
            "eval_k": int(cfg.topk),
            "Certified": int(mix_cert),
            "ndcg": float(best_mix["test_ndcg"]),
            "recall": float(best_mix["test_recall"]),
            "Val.UCB": float(best_mix["val_ucb"]),
            "Test.qleak": float(best_mix["test_qleak"]),
            "Stat.AUC": maybe_float(mix_auc),
            "LLM.AUC": llm_summary_by_policy["RA+Mix"]["LLM.AUC"],
            "LLM.SingleAcc": llm_summary_by_policy["RA+Mix"]["LLM.SingleAcc"],
            "LLM.PairAcc": llm_summary_by_policy["RA+Mix"]["LLM.PairAcc"],
            "Monitor.Score": llm_summary_by_policy["RA+Mix"]["Monitor.Score"],
            "StateReveal.Score": llm_summary_by_policy["RA+Mix"]["StateReveal.Score"],
            "PrivacyInvasive.Score": llm_summary_by_policy["RA+Mix"]["PrivacyInvasive.Score"],
            "selected_tau": float(selected_tau_mix),
            "selected_gamma_iter": None,
            "selected_gamma_l1": float(best_mix["gamma_l1"]),
            "selected_alpha": float(best_mix_alpha),
            "selected_lambda": float(best_mix_lam),
            "Val.UCB.add": float(best_mix["val_ucb_add"]),
            "Test.qleak.add": float(best_mix["test_qleak_add"]),
            "state_family": str(cfg.state_family),
            "state_eta": float(cfg.state_eta),
            "policy_eta": float(get_policy_eta(cfg)),
        },
    ]

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(seed_dir, "table1_seed_fullgamma.csv"), index=False)

    with open(os.path.join(seed_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "ranker_name": active_ranker,
                "main_metric_family": "full_G_main_paper",
                "diag_metric_family": "smooth_diag",
                "figure3_mode": getattr(cfg, "figure3_mode", "scalar_lambda"),
                "state_family": str(cfg.state_family),
                "state_eta": float(cfg.state_eta),
                "policy_eta": float(get_policy_eta(cfg)),
                "eval_k": int(cfg.topk),
                "num_dev_users": len(dev_users),
                "num_val_users": len(val_users),
                "num_test_users": len(test_users),
                "configured_tau": float(cfg.tau),
                "tau_selection_mode": getattr(cfg, "tau_selection_mode", "fixed"),
                "selected_tau_ra": float(selected_tau_ra),
                "selected_tau_mix": float(selected_tau_mix),
                "best_ra_candidate_key": best_ra_key,
                "best_ra_gamma_iter": None,
                "best_ra_gamma_l1": float(best_ra["gamma_l1"]),
                "best_ra_lambda": float(best_ra_lam),
                "best_ra_certified": ra_cert,
                "best_mix_candidate_key": best_mix_key,
                "best_mix_gamma_iter": None,
                "best_mix_gamma_l1": float(best_mix["gamma_l1"]),
                "best_mix_lambda": float(best_mix_lam),
                "best_mix_alpha": float(best_mix_alpha),
                "best_mix_certified": mix_cert,
                "rad_val_main": rad_val_main,
                "rad_val_cert": rad_val_cert,
                "num_full_queries": num_full_queries,
                "num_cert_queries": num_cert_queries,
                "num_cert_base_features": int(item_feat_mat.shape[1]),
                "cert_base_feature_names": cert_base_feat_names,
                "num_candidates_total": num_candidates_total,
                "ra_top_val_query_main": ra_top_main["top_query"],
                "ra_top_val_abs_main": ra_top_main["top_abs_diff"],
                "mix_top_val_query_main": mix_top_main["top_query"],
                "mix_top_val_abs_main": mix_top_main["top_abs_diff"],
                "aware_val_diag_qleak": float(aware_val_diag["qleak"]),
                "aware_test_diag_qleak": float(aware_test_diag["qleak"]),
                "heldout_alignment": heldout_alignment,
                "ra_auc_cert_only": ra_auc_cert_only,
                "ra_auc_full_features": ra_auc,
                "mix_auc_cert_only": mix_auc_cert,
                "mix_auc_full_features": mix_auc,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(runtime_rows).to_csv(os.path.join(seed_dir, "table6_runtime.csv"), index=False)
    return final_rows
def get_ranker_variant_cfg(base_cfg: Config, variant_name: str) -> Config:
    if variant_name == "default":
        return replace(base_cfg)

    if variant_name == "small":
        return replace(
            base_cfg,
            embed_dim=24,
            epochs=max(6, base_cfg.epochs - 2),
            lr=0.05,
            l2=0.003,
        )

    if variant_name == "large":
        return replace(
            base_cfg,
            embed_dim=96,
            epochs=max(15, base_cfg.epochs),
            lr=0.03,
            l2=0.001,
        )

    raise ValueError(f"Unknown ranker robustness variant: {variant_name}")

def run_ranker_robustness(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "ranker_robustness")
    ensure_dir(sweep_root)

    summary_rows = []

    for ranker_name in getattr(base_cfg, "ranker_robustness_models", ("bprmf", "lightgcn", "sasrec")):
        cfg_var = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, ranker_name),
            seeds=base_cfg.sweep_seeds,
            llm_enabled=base_cfg.ranker_robustness_llm_enabled,
            run_tau_sweep=False,
            run_eta_sweep=False,
            run_k_sweep=False,
            run_ranker_robustness=False,
            ranker_name=ranker_name,
        )

        out = run_experiment_bundle(cfg_var, data_dict, bundle_name=f"ranker={ranker_name}")
        agg_df = out["agg_df"].copy()
        agg_df["ranker_model"] = ranker_name
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(sweep_root, "ranker_robustness_summary.csv"),
            index=False,
        )


def run_experiment_bundle(cfg: Config, data_dict, bundle_name: str = "main"):
    ensure_dir(cfg.out_dir)

    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_rows = []
    for seed in cfg.seeds:
        rows = run_one_seed(cfg, data_dict, seed, ranker_name=getattr(cfg, "ranker_name", "bprmf"))
        all_rows.extend(rows)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(cfg.out_dir, "table1_all_seeds_full.csv"), index=False)

    agg_df = aggregate_results(all_df)
    agg_df.to_csv(os.path.join(cfg.out_dir, "table1_aggregate_full.csv"), index=False)

    pretty_df = prettify_aggregate(agg_df)
    pretty_df.to_csv(os.path.join(cfg.out_dir, "table1_pretty_full.csv"), index=False)

    selection_df = build_selection_summary(all_df)
    selection_df.to_csv(os.path.join(cfg.out_dir, "selection_by_seed.csv"), index=False)

    aggregate_auxiliary_outputs(cfg.out_dir)

    print(f"\n=== [{bundle_name}] Table 1 (aggregate over seeds) ===")
    print(pretty_df.to_string(index=False))

    print(f"\n=== [{bundle_name}] Selected gamma/alpha by seed ===")
    if len(selection_df) > 0:
        print(selection_df.to_string(index=False))
    else:
        print("(no RA / RA+Mix rows found)")

    return {
        "all_df": all_df,
        "agg_df": agg_df,
        "pretty_df": pretty_df,
        "selection_df": selection_df,
    }
def run_state_eta_sweep(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "state_eta_sweep")
    ensure_dir(sweep_root)

    summary_rows = []

    for state_eta in base_cfg.eta_grid:
        cfg_eta = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, f"state_eta_{float(state_eta):.3f}"),
            state_eta=float(state_eta),
            policy_eta=None,   # tie mitigation strength to hidden-state strength
            seeds=base_cfg.sweep_seeds,
            llm_enabled=base_cfg.sweep_llm_enabled,
            run_tau_sweep=False,
            run_eta_sweep=False,
            run_k_sweep=False,
            run_ranker_robustness=False,
            run_state_family_sweep=False,
        )
        out = run_experiment_bundle(cfg_eta, data_dict, bundle_name=f"state_eta={float(state_eta):.3f}")
        agg_df = out["agg_df"].copy()
        agg_df["state_eta"] = float(state_eta)
        agg_df["policy_eta"] = float(get_policy_eta(cfg_eta))
        agg_df["state_family"] = str(cfg_eta.state_family)
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(sweep_root, "state_eta_sweep_summary.csv"),
            index=False,
        )
def run_state_family_sweep(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "state_family_sweep")
    ensure_dir(sweep_root)

    summary_rows = []

    for fam in base_cfg.state_family_grid:
        cfg_f = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, f"state_family_{fam}"),
            state_family=str(fam),
            seeds=base_cfg.sweep_seeds,
            llm_enabled=False,
            run_tau_sweep=False,
            run_eta_sweep=False,
            run_k_sweep=False,
            run_ranker_robustness=False,
            run_state_family_sweep=False,
        )
        out = run_experiment_bundle(cfg_f, data_dict, bundle_name=f"state_family={fam}")
        agg_df = out["agg_df"].copy()
        agg_df["state_family"] = str(fam)
        agg_df["state_eta"] = float(cfg_f.state_eta)
        agg_df["policy_eta"] = float(get_policy_eta(cfg_f))
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(sweep_root, "state_family_sweep_summary.csv"),
            index=False,
        )
def run_k_sweep(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "k_sweep")
    ensure_dir(sweep_root)

    summary_rows = []

    for k in base_cfg.k_grid:
        cfg_k = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, f"k_{int(k)}"),
            topk=int(k),
            seeds=base_cfg.sweep_seeds,
            llm_enabled=False,
        )
        out = run_experiment_bundle(cfg_k, data_dict, bundle_name=f"k={int(k)}")
        agg_df = out["agg_df"].copy()
        agg_df["topk"] = int(k)
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(sweep_root, "k_sweep_summary.csv"),
            index=False,
        )

def get_ranker_variant_cfg(base_cfg: Config, variant_name: str) -> Config:
    if variant_name == "default":
        return replace(base_cfg)

    if variant_name == "small":
        return replace(
            base_cfg,
            embed_dim=24,
            epochs=max(6, base_cfg.epochs - 2),
            lr=0.05,
            l2=0.003,
        )

    if variant_name == "large":
        return replace(
            base_cfg,
            embed_dim=96,
            epochs=max(15, base_cfg.epochs),
            lr=0.03,
            l2=0.001,
        )

    raise ValueError(f"Unknown ranker robustness variant: {variant_name}")

def run_ranker_robustness(base_cfg: Config, data_dict):
    sweep_root = os.path.join(base_cfg.out_dir, "ranker_robustness")
    ensure_dir(sweep_root)

    summary_rows = []

    for ranker_name in getattr(base_cfg, "ranker_robustness_models", ("bprmf", "lightgcn", "sasrec")):
        cfg_var = replace(
            base_cfg,
            out_dir=os.path.join(sweep_root, ranker_name),
            seeds=base_cfg.sweep_seeds,
            llm_enabled=base_cfg.ranker_robustness_llm_enabled,
            run_tau_sweep=False,
            run_eta_sweep=False,
            run_k_sweep=False,
            run_ranker_robustness=False,
            ranker_name=ranker_name,
        )

        out = run_experiment_bundle(cfg_var, data_dict, bundle_name=f"ranker={ranker_name}")
        agg_df = out["agg_df"].copy()
        agg_df["ranker_model"] = ranker_name
        summary_rows.append(agg_df)

    if len(summary_rows) > 0:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(sweep_root, "ranker_robustness_summary.csv"),
            index=False,
        )