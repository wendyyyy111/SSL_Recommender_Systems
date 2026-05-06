import argparse
import torch
from dataclasses import replace

from src.config import Config, resolve_device
from src.data import load_ml1m
from src.experiment import (
    run_experiment_bundle,
    run_tau_sweep,
    run_state_eta_sweep,
    run_k_sweep,
    run_ranker_robustness,
    run_state_family_sweep,
)
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if torch.cuda.is_available() and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=device,
    )

    if cfg.minimal_run:
        cfg = replace(
            cfg,
            llm_enabled=False if cfg.disable_llm_in_minimal else cfg.llm_enabled,
            run_tau_sweep=False if cfg.disable_sweeps_in_minimal else cfg.run_tau_sweep,
            run_eta_sweep=False if cfg.disable_sweeps_in_minimal else cfg.run_eta_sweep,
            run_k_sweep=False if cfg.disable_sweeps_in_minimal else cfg.run_k_sweep,
            run_ranker_robustness=False if cfg.disable_sweeps_in_minimal else cfg.run_ranker_robustness,
        )

    ensure_dir(cfg.out_dir)
    data_dict = load_ml1m(cfg.data_dir)

    run_experiment_bundle(cfg, data_dict, bundle_name="main")

    if cfg.run_tau_sweep:
        run_tau_sweep(cfg, data_dict)

    if cfg.run_eta_sweep:
        run_state_eta_sweep(cfg, data_dict)

    if cfg.run_k_sweep:
        run_k_sweep(cfg, data_dict)

    if cfg.run_ranker_robustness:
        run_ranker_robustness(cfg, data_dict)

    if cfg.run_state_family_sweep:
        run_state_family_sweep(cfg, data_dict)

    print(f"\nSaved outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    main()