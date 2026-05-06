from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .queries import build_cert_query_vector, transcript_stats
def safe_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(y_true) == 0:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan
def safe_balanced_accuracy_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(y_true) == 0:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        if np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
            pred = (scores >= 0.5).astype(int)
        else:
            pred = (scores >= 0.0).astype(int)
        return float(balanced_accuracy_score(y_true, pred))
    except Exception:
        return np.nan


def build_attacker_feature_vector(
    topk_items: np.ndarray,
    user_hist_genre: np.ndarray,
    item_genre_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
):
    s = transcript_stats(topk_items, item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx)
    E = s["E"]
    Ebar = s["Ebar"]
    g_cent = (1.0 + E - user_hist_genre) / 2.0

    feat = np.concatenate(
        [
            E,
            Ebar,
            g_cent,
            np.array([s["entropy"]], dtype=np.float32),
            np.array([s["concentration"]], dtype=np.float32),
            np.array([s["bucket0_mass"], s["bucket1_mass"]], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return feat
def build_attacker_dataset_cert_only(
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    item_feat_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
):
    X, y = [], []
    for u in users:
        q0 = build_cert_query_vector(topk_z0[u], item_feat_mat, rank_w, rank_w_sum)
        q1 = build_cert_query_vector(topk_z1[u], item_feat_mat, rank_w, rank_w_sum)
        X.append(q0)
        y.append(0)
        X.append(q1)
        y.append(1)
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)

def build_attacker_dataset_cert_only(
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    item_feat_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
):
    X, y = [], []
    for u in users:
        q0 = build_cert_query_vector(topk_z0[u], item_feat_mat, rank_w, rank_w_sum)
        q1 = build_cert_query_vector(topk_z1[u], item_feat_mat, rank_w, rank_w_sum)
        X.append(q0)
        y.append(0)
        X.append(q1)
        y.append(1)
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)
def fit_and_eval_logreg_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    score = clf.predict_proba(X_test)[:, 1]
    return {
        "auc": safe_auc_from_scores(y_test, score),
        "ba": safe_balanced_accuracy_from_scores(y_test, score),
    }


def fit_and_eval_rf_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_test)[:, 1]
    else:
        score = clf.predict(X_test)
    return {
        "auc": safe_auc_from_scores(y_test, score),
        "ba": safe_balanced_accuracy_from_scores(y_test, score),
    }



def fit_and_eval_gbdt_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=0,
    )
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X_test)
    else:
        score = clf.predict(X_test)

    return {
        "auc": safe_auc_from_scores(y_test, score),
        "ba": safe_balanced_accuracy_from_scores(y_test, score),
    }


def fit_and_eval_mlp_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=128,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_test)[:, 1]
    else:
        score = clf.predict(X_test)

    return {
        "auc": safe_auc_from_scores(y_test, score),
        "ba": safe_balanced_accuracy_from_scores(y_test, score),
    }


def eval_policy_auc_suite(
    val_users: List[int],
    test_users: List[int],
    topk_val_z0: Dict[int, np.ndarray],
    topk_val_z1: Dict[int, np.ndarray],
    topk_test_z0: Dict[int, np.ndarray],
    topk_test_z1: Dict[int, np.ndarray],
    item_genre_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
    cert_only: bool = False,
    item_feat_mat: Optional[np.ndarray] = None,
    h_user: Optional[Dict[int, np.ndarray]] = None,
):
    if cert_only:
        feat_mat = item_genre_mat if item_feat_mat is None else item_feat_mat
        X_train, y_train = build_attacker_dataset_cert_only(
            val_users, topk_val_z0, topk_val_z1, feat_mat, rank_w, rank_w_sum
        )
        X_test, y_test = build_attacker_dataset_cert_only(
            test_users, topk_test_z0, topk_test_z1, feat_mat, rank_w, rank_w_sum
        )
    else:
        if h_user is None:
            raise ValueError("h_user must be provided when cert_only=False.")
        X_train, y_train = build_attacker_dataset(
            val_users, topk_val_z0, topk_val_z1,
            h_user, item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx
        )
        X_test, y_test = build_attacker_dataset(
            test_users, topk_test_z0, topk_test_z1,
            h_user, item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx
        )

    lg = fit_and_eval_logreg_metrics(X_train, y_train, X_test, y_test)
    rf = fit_and_eval_rf_metrics(X_train, y_train, X_test, y_test)
    gb = fit_and_eval_gbdt_metrics(X_train, y_train, X_test, y_test)
    mlp = fit_and_eval_mlp_metrics(X_train, y_train, X_test, y_test)

    out = {
        "logreg_auc": lg["auc"],
        "logreg_ba": lg["ba"],
        "rf_auc": rf["auc"],
        "rf_ba": rf["ba"],
        "gbdt_auc": gb["auc"],
        "gbdt_ba": gb["ba"],
        "mlp_auc": mlp["auc"],
        "mlp_ba": mlp["ba"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
        "cert_only": int(cert_only),
    }
    return out

def run_statistical_attacker_suite(
    cfg: Config,
    seed: int,
    seed_dir: str,
    val_users: List[int],
    test_users: List[int],
    topk_aware_z0: Dict[int, np.ndarray],
    topk_aware_z1: Dict[int, np.ndarray],
    topk_agn: Dict[int, np.ndarray],
    topk_ra_sel_z0: Dict[int, np.ndarray],
    topk_ra_sel_z1: Dict[int, np.ndarray],
    best_ra_cand: Dict[str, Any],
    best_mix_cand: Dict[str, Any],
    best_mix_alpha: float,
    item_genre_mat: np.ndarray,
    rank_w: np.ndarray,
    rank_w_sum: float,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
    h_user: Dict[int, np.ndarray],
    heldout: Dict[int, int],
    item_feat_mat: np.ndarray,
    agn_val_G: Dict[str, Any],
    agn_test_G: Dict[str, Any],
    heldout_z1: Optional[Dict[int, int]] = None,

):
    aware_auc_suite = eval_policy_auc_suite(
        val_users, test_users,
        topk_aware_z0, topk_aware_z1,
        topk_aware_z0, topk_aware_z1,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
        cert_only=False,
        h_user=h_user,
    )

    agn_auc_suite = eval_policy_auc_suite(
        val_users, test_users,
        topk_agn, topk_agn,
        topk_agn, topk_agn,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
        cert_only=False,
        h_user=h_user,
    )

    ra_auc_suite = eval_policy_auc_suite(
        val_users, test_users,
        topk_ra_sel_z0, topk_ra_sel_z1,
        topk_ra_sel_z0, topk_ra_sel_z1,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
        cert_only=False,
        h_user=h_user,
    )

    ra_auc_cert_suite = eval_policy_auc_suite(
        val_users, test_users,
        topk_ra_sel_z0, topk_ra_sel_z1,
        topk_ra_sel_z0, topk_ra_sel_z1,
        item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
        cert_only=True,
        item_feat_mat=item_feat_mat,
    )

    mix_logreg_aucs, mix_rf_aucs, mix_gbdt_aucs, mix_mlp_aucs = [], [], [], []
    mix_logreg_bas, mix_rf_bas, mix_gbdt_bas, mix_mlp_bas = [], [], [], []
    mix_aucs_cert = []
    mix_bas_cert = []

    for mc in range(cfg.attacker_mix_mc):
        tr_z0, tr_z1 = sample_mix_transcripts(
            val_users,
            topk_agn=topk_agn,
            topk_ra_z0=best_mix_cand["topk_z0"],
            topk_ra_z1=best_mix_cand["topk_z1"],
            alpha=best_mix_alpha,
            seed=seed * 10000 + 100 * mc + 1,
            shared_coin=cfg.mix_shared_coin_in_audit,
        )
        te_z0, te_z1 = sample_mix_transcripts(
            test_users,
            topk_agn=topk_agn,
            topk_ra_z0=best_mix_cand["topk_z0"],
            topk_ra_z1=best_mix_cand["topk_z1"],
            alpha=best_mix_alpha,
            seed=seed * 10000 + 100 * mc + 2,
            shared_coin=cfg.mix_shared_coin_in_audit,
        )

        mix_suite = eval_policy_auc_suite(
            val_users, test_users,
            tr_z0, tr_z1,
            te_z0, te_z1,
            item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
            cert_only=False,
            h_user=h_user,
        )
        mix_logreg_aucs.append(mix_suite["logreg_auc"])
        mix_rf_aucs.append(mix_suite["rf_auc"])
        mix_gbdt_aucs.append(mix_suite["gbdt_auc"])
        mix_mlp_aucs.append(mix_suite["mlp_auc"])
        mix_logreg_bas.append(mix_suite["logreg_ba"])
        mix_rf_bas.append(mix_suite["rf_ba"])
        mix_gbdt_bas.append(mix_suite["gbdt_ba"])
        mix_mlp_bas.append(mix_suite["mlp_ba"])

        mix_cert_suite = eval_policy_auc_suite(
            val_users, test_users,
            tr_z0, tr_z1,
            te_z0, te_z1,
            item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
            cert_only=True,
            item_feat_mat=item_feat_mat,
        )
        mix_aucs_cert.append(mix_cert_suite["logreg_auc"])
        mix_bas_cert.append(mix_cert_suite["logreg_ba"])

    attacker_breakdown_rows = [
        {
            "policy": "State-aware",
            "logreg_auc": aware_auc_suite["logreg_auc"],
            "logreg_ba": aware_auc_suite["logreg_ba"],
            "rf_auc": aware_auc_suite["rf_auc"],
            "rf_ba": aware_auc_suite["rf_ba"],
            "gbdt_auc": aware_auc_suite["gbdt_auc"],
            "gbdt_ba": aware_auc_suite["gbdt_ba"],
            "mlp_auc": aware_auc_suite["mlp_auc"],
            "mlp_ba": aware_auc_suite["mlp_ba"],
        },
        {
            "policy": "State-independent",
            "logreg_auc": agn_auc_suite["logreg_auc"],
            "logreg_ba": agn_auc_suite["logreg_ba"],
            "rf_auc": agn_auc_suite["rf_auc"],
            "rf_ba": agn_auc_suite["rf_ba"],
            "gbdt_auc": agn_auc_suite["gbdt_auc"],
            "gbdt_ba": agn_auc_suite["gbdt_ba"],
            "mlp_auc": agn_auc_suite["mlp_auc"],
            "mlp_ba": agn_auc_suite["mlp_ba"],
        },
        {
            "policy": "RA",
            "logreg_auc": ra_auc_suite["logreg_auc"],
            "logreg_ba": ra_auc_suite["logreg_ba"],
            "rf_auc": ra_auc_suite["rf_auc"],
            "rf_ba": ra_auc_suite["rf_ba"],
            "gbdt_auc": ra_auc_suite["gbdt_auc"],
            "gbdt_ba": ra_auc_suite["gbdt_ba"],
            "mlp_auc": ra_auc_suite["mlp_auc"],
            "mlp_ba": ra_auc_suite["mlp_ba"],
        },
        {
            "policy": "RA+Mix",
            "logreg_auc": float(np.nanmean(mix_logreg_aucs)) if len(mix_logreg_aucs) > 0 else np.nan,
            "logreg_ba": float(np.nanmean(mix_logreg_bas)) if len(mix_logreg_bas) > 0 else np.nan,
            "rf_auc": float(np.nanmean(mix_rf_aucs)) if len(mix_rf_aucs) > 0 else np.nan,
            "rf_ba": float(np.nanmean(mix_rf_bas)) if len(mix_rf_bas) > 0 else np.nan,
            "gbdt_auc": float(np.nanmean(mix_gbdt_aucs)) if len(mix_gbdt_aucs) > 0 else np.nan,
            "gbdt_ba": float(np.nanmean(mix_gbdt_bas)) if len(mix_gbdt_bas) > 0 else np.nan,
            "mlp_auc": float(np.nanmean(mix_mlp_aucs)) if len(mix_mlp_aucs) > 0 else np.nan,
            "mlp_ba": float(np.nanmean(mix_mlp_bas)) if len(mix_mlp_bas) > 0 else np.nan,
        },
    ]
    pd.DataFrame(attacker_breakdown_rows).to_csv(
        os.path.join(seed_dir, "table2_attacker_auc_suite.csv"), index=False
    )

    mix_val_diag_list = []
    mix_test_diag_list = []
    for mc in range(max(1, cfg.attacker_mix_mc)):
        mz0_val, mz1_val = sample_mix_transcripts(
            val_users,
            topk_agn=topk_agn,
            topk_ra_z0=best_mix_cand["topk_z0"],
            topk_ra_z1=best_mix_cand["topk_z1"],
            alpha=best_mix_alpha,
            seed=seed * 50000 + 100 * mc + 1,
            shared_coin=cfg.mix_shared_coin_in_audit,
        )
        mz0_te, mz1_te = sample_mix_transcripts(
            test_users,
            topk_agn=topk_agn,
            topk_ra_z0=best_mix_cand["topk_z0"],
            topk_ra_z1=best_mix_cand["topk_z1"],
            alpha=best_mix_alpha,
            seed=seed * 50000 + 100 * mc + 2,
            shared_coin=cfg.mix_shared_coin_in_audit,
        )
        mix_val_diag_list.append(
            evaluate_policy_metrics(
                val_users, mz0_val, mz1_val, heldout, h_user,
                item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
                heldout_z1=heldout_z1
            )
        )
        mix_test_diag_list.append(
            evaluate_policy_metrics(
                test_users, mz0_te, mz1_te, heldout, h_user,
                item_genre_mat, rank_w, rank_w_sum, bucket0_idx, bucket1_idx,
                heldout_z1=heldout_z1
            )
        )

    mix_val_diag_qleak = float(np.mean([x["qleak"] for x in mix_val_diag_list]))
    mix_test_diag_qleak = float(np.mean([x["qleak"] for x in mix_test_diag_list]))

    diag_rows = [
        {"policy": "State-aware", "split": "val", "diag_qleak": np.nan},
        {"policy": "State-aware", "split": "test", "diag_qleak": np.nan},
        {"policy": "State-independent", "split": "val", "diag_qleak": agn_val_G["qleak"]},
        {"policy": "State-independent", "split": "test", "diag_qleak": agn_test_G["qleak"]},
        {"policy": "RA", "split": "val", "diag_qleak": best_ra_cand["val_metrics_G"]["qleak"]},
        {"policy": "RA", "split": "test", "diag_qleak": best_ra_cand["test_metrics_G"]["qleak"]},
        {"policy": "RA+Mix", "split": "val", "diag_qleak": mix_val_diag_qleak},
        {"policy": "RA+Mix", "split": "test", "diag_qleak": mix_test_diag_qleak},
    ]
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(seed_dir, "diagnostic_nonadditive_qleak.csv"), index=False
    )

    return {
        "aware_auc": aware_auc_suite["logreg_auc"],
        "agn_auc": agn_auc_suite["logreg_auc"],
        "ra_auc": ra_auc_suite["logreg_auc"],
        "ra_auc_cert_only": ra_auc_cert_suite["logreg_auc"],
        "mix_auc": float(np.nanmean(mix_logreg_aucs)) if len(mix_logreg_aucs) > 0 else np.nan,
        "mix_auc_cert": float(np.nanmean(mix_aucs_cert)) if len(mix_aucs_cert) > 0 else np.nan,
    }
