from dataclasses import dataclass
from typing import Tuple, Optional
import torch


POLICY_ORDER = ["State-aware", "State-independent", "RA", "RA+Mix"]


@dataclass
class Config:
    data_dir: str
    out_dir: str
    device: str = "cuda"

    ranker_name: str = "sasrec"
    figure3_mode: str = "scalar_lambda"

    minimal_run: bool = False
    use_paired_state_targets: bool = True
    holdout_window: int = 4
    min_train_prefix: int = 5
    fast_attacker_only_logreg: bool = True
    disable_llm_in_minimal: bool = True
    disable_sweeps_in_minimal: bool = True
    tau_selection_mode: str = "fixed"

    debug_strict: bool = False
    debug_verbose: bool = False

    scalar_risk_mode: str = "dev_leak_linear"
    scalar_risk_center: bool = True
    scalar_risk_scale_by_std: bool = True
    scalar_risk_clip_z: float = 3.0

    rerank_normalize_scores: bool = True
    rerank_score_eps: float = 1e-8

    stateaware_rerank_M: int = 300

    embed_dim: int = 48
    lr: float = 0.05
    l2: float = 0.002
    epochs: int = 10
    samples_per_user_per_epoch: int = 35
    batch_size: int = 8192

    bonus_center: bool = True
    bonus_scale_by_std: bool = True
    bonus_clip_z: float = 3.0
    bonus_clip_abs: float = 0.20
    bonus_eps: float = 1e-8

    bucket0: Tuple[str, ...] = ("Romance", "Drama")
    bucket1: Tuple[str, ...] = ("Action", "Thriller")

    lambda_grid: Tuple[float, ...] = (
        0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.07, 0.10, 0.12, 0.15, 0.18, 0.20,
        0.25, 0.30, 0.40, 0.50, 0.70, 1.0
    )

    alpha_grid: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.8, 1.0)
    mix_shared_coin_in_audit: bool = True

    delta: float = 0.05
    tau: float = 0.18

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    sweep_seeds: Tuple[int, ...] = (0, 1, 2)

    attacker_mix_mc: int = 3

    llm_enabled: bool = False
    llm_registry_path: str = "llm_registry.json"
    llm_cache_path: str = "llm_cache.jsonl"
    llm_model_name: str = "openai_gpt4o"
    llm_temperature: float = 0.0
    llm_timeout_sec: int = 120
    llm_max_tokens_single: int = 256
    llm_max_tokens_pair: int = 256
    llm_list_k: int = 10
    llm_max_users: Optional[int] = 100
    llm_mix_mc: int = 3
    llm_personas: Tuple[str, ...] = (
        "Casual observer",
        "Privacy-aware user",
        "Domain-informed observer",
        "Recommender analyst",
        "Zero-trust auditor",
    )
    llm_bootstrap_n: int = 1000
    llm_bootstrap_alpha: float = 0.05

    state_desc_0: str = "a short-term preference for romance/drama content"
    state_desc_1: str = "a short-term preference for action/thriller content"

    topk: int = 50
    dev_ratio: float = 0.12
    val_ratio: float = 0.78
    test_ratio: float = 0.10

    gamma_extra_scales: Tuple[float, ...] = ()
    gamma_topm_values: Tuple[int, ...] = ()
    gamma_steps: int = 10
    gamma_step_size: float = 0.01
    gamma_clip: float = 3.0
    gamma_snapshot_every: int = 2

    state_family: str = "semantic_linear"
    state_eta: float = 0.01
    policy_eta: Optional[float] = None

    run_state_family_sweep: bool = True
    state_family_grid: Tuple[str, ...] = ("semantic_linear", "sparse_bucket", "history_conditioned")

    sparse_bucket0: Tuple[str, ...] = ("Romance",)
    sparse_bucket1: Tuple[str, ...] = ("Action",)

    history_state_mix_weight: float = 0.5

    run_tau_sweep: bool = True
    tau_grid: Tuple[float, ...] = (0.10, 0.11, 0.12, 0.13)

    run_eta_sweep: bool = False
    eta_grid: Tuple[float, ...] = (0.02, 0.03, 0.04, 0.06, 0.08)

    run_k_sweep: bool = False
    k_grid: Tuple[int, ...] = (10, 20, 50, 100)

    run_ranker_robustness: bool = False
    ranker_robustness_models: Tuple[str, ...] = ("bprmf", "lightgcn", "sasrec")
    ranker_robustness_llm_enabled: bool = False
    sweep_llm_enabled: bool = False

    policy_runtime_repeats: int = 1

    use_exposure_risk_feature: bool = True
    use_context_stratified_risk: bool = False
    risk_rank_strata: Tuple[Tuple[int, int], ...] = ()
    risk_smoothing: float = 1.0
    risk_clip_B: float = 1.0

    run_stateblind_risk_baseline: bool = False
    stateblind_risk_beta_grid: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)

    lightgcn_layers: int = 3
    lightgcn_lr: float = 1e-3
    lightgcn_l2: float = 1e-4
    lightgcn_epochs: int = 10

    sasrec_max_seq_len: int = 50
    sasrec_num_heads: int = 2
    sasrec_num_layers: int = 2
    sasrec_dropout: float = 0.2
    sasrec_lr: float = 1e-3
    sasrec_l2: float = 1e-4
    sasrec_epochs: int = 10


def get_policy_eta(cfg: Config) -> float:
    return float(cfg.state_eta if cfg.policy_eta is None else cfg.policy_eta)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg