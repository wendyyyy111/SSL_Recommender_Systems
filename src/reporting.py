import os
import pandas as pd
import numpy as np

from .config import POLICY_ORDER
from .utils import fmt_mean_std, fmt_percent


def aggregate_results(all_df: pd.DataFrame):
    df = all_df.copy()

    numeric_cols = [
        "eval_k",
        "Certified",
        "ndcg",
        "recall",
        "Val.UCB",
        "Test.qleak",
        "Stat.AUC",
        "LLM.AUC",
        "LLM.SingleAcc",
        "LLM.PairAcc",
        "Monitor.Score",
        "StateReveal.Score",
        "PrivacyInvasive.Score",
        "selected_gamma_iter",
        "selected_gamma_l1",
        "selected_alpha",
        "selected_lambda",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_spec = {}
    for c in numeric_cols:
        if c not in df.columns:
            continue
        if c == "eval_k":
            agg_spec[c] = ["mean"]
        else:
            agg_spec[c] = ["mean", "std"]

    agg_df = df.groupby("policy", as_index=False).agg(agg_spec)

    flat_cols = []
    for c in agg_df.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    agg_df.columns = flat_cols

    if "policy_" in agg_df.columns:
        agg_df = agg_df.rename(columns={"policy_": "policy"})

    if "eval_k_mean" in agg_df.columns:
        agg_df["eval_k"] = agg_df["eval_k_mean"].round().astype("Int64")
        agg_df = agg_df.drop(columns=["eval_k_mean"])

    order_map = {p: i for i, p in enumerate(POLICY_ORDER)}
    agg_df["policy_order"] = agg_df["policy"].map(order_map)
    agg_df = agg_df.sort_values("policy_order").drop(columns=["policy_order"]).reset_index(drop=True)
    return agg_df


def prettify_aggregate(agg_df: pd.DataFrame):
    rows = []
    for _, r in agg_df.iterrows():
        k = int(r["eval_k"]) if "eval_k" in agg_df.columns and pd.notna(r["eval_k"]) else "K"
        rows.append(
            {
                "Policy": r["policy"],
                "Certified rate": fmt_percent(r["Certified_mean"], digits=1),
                f"NDCG@{k} ↑": fmt_mean_std(r["ndcg_mean"], r["ndcg_std"]),
                f"Recall@{k} ↑": fmt_mean_std(r["recall_mean"], r["recall_std"]),
                "Val.UCB ↓": fmt_mean_std(r["Val.UCB_mean"], r["Val.UCB_std"]),
                "Test.qleak ↓": fmt_mean_std(r["Test.qleak_mean"], r["Test.qleak_std"]),
                "Stat.AUC → .5": fmt_mean_std(r["Stat.AUC_mean"], r["Stat.AUC_std"]),
                "LLM.AUC → .5": fmt_mean_std(r["LLM.AUC_mean"], r["LLM.AUC_std"]),
                "LLM.PairAcc → .5": fmt_mean_std(r["LLM.PairAcc_mean"], r["LLM.PairAcc_std"]),
                "Monitor.Score ↓": fmt_mean_std(r["Monitor.Score_mean"], r["Monitor.Score_std"]),
                "Sel. λ": fmt_mean_std(r["selected_lambda_mean"], r["selected_lambda_std"], digits=4),
                "Sel. α": fmt_mean_std(r["selected_alpha_mean"], r["selected_alpha_std"], digits=4),
            }
        )
    return pd.DataFrame(rows)


def build_selection_summary(all_df: pd.DataFrame):
    cols = ["seed", "policy", "Certified", "selected_gamma_iter", "selected_gamma_l1", "selected_alpha", "selected_lambda"]
    out = all_df[cols].copy()
    out = out[out["policy"].isin(["RA", "RA+Mix"])].reset_index(drop=True)
    return out


def aggregate_auxiliary_outputs(out_dir: str):
    seed_dirs = []
    for name in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, name)
        if os.path.isdir(full) and name.startswith("seed_"):
            seed_dirs.append(full)

    def _flatten_cols(df):
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                if c[1] == "":
                    new_cols.append(c[0])
                else:
                    new_cols.append(f"{c[0]}_{c[1]}")
            else:
                new_cols.append(c)
        df.columns = new_cols
        return df

    attacker_frames = []
    runtime_frames = []

    for sd in seed_dirs:
        seed_name = os.path.basename(sd)
        try:
            seed = int(seed_name.split("_", 1)[1])
        except Exception:
            seed = None

        p1 = os.path.join(sd, "table2_attacker_auc_suite.csv")
        if os.path.exists(p1):
            df = pd.read_csv(p1)
            df["seed"] = seed
            attacker_frames.append(df)

        p2 = os.path.join(sd, "table6_runtime.csv")
        if os.path.exists(p2):
            df = pd.read_csv(p2)
            df["seed"] = seed
            runtime_frames.append(df)

    if len(attacker_frames) > 0:
        all_att = pd.concat(attacker_frames, axis=0, ignore_index=True)
        metric_cols = [c for c in all_att.columns if c not in {"policy", "seed"}]
        agg = all_att.groupby("policy", as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
        agg = _flatten_cols(agg)
        if "policy_" in agg.columns:
            agg = agg.rename(columns={"policy_": "policy"})
        agg.to_csv(os.path.join(out_dir, "table2_attacker_auc_suite_aggregate.csv"), index=False)

    if len(runtime_frames) > 0:
        all_rt = pd.concat(runtime_frames, axis=0, ignore_index=True)

        pol = all_rt[all_rt["runtime_type"] == "policy_level"].copy()
        if len(pol) > 0:
            metric_cols = ["seconds", "mean_ms_per_query"]
            agg_pol = pol.groupby(["policy"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
            agg_pol = _flatten_cols(agg_pol)
            if "policy_" in agg_pol.columns:
                agg_pol = agg_pol.rename(columns={"policy_": "policy"})
            agg_pol.to_csv(os.path.join(out_dir, "table6_runtime_policy_aggregate.csv"), index=False)

        stg = all_rt[all_rt["runtime_type"] == "stage_level"].copy()
        if len(stg) > 0:
            metric_cols = ["seconds"]
            agg_stg = stg.groupby(["stage"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
            agg_stg = _flatten_cols(agg_stg)
            if "stage_" in agg_stg.columns:
                agg_stg = agg_stg.rename(columns={"stage_": "stage"})
            agg_stg.to_csv(os.path.join(out_dir, "table6_runtime_stage_aggregate.csv"), index=False)