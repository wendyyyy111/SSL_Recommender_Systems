# SSL_Recommender_Systems


A modular research codebase for auditing privacy leakage in recommender systems on the MovieLens-1M dataset, with support for multiple base recommenders, privacy-aware re-ranking policies, certified additive leakage analysis, broader query-family evaluation, statistical attackers, and optional LLM-based auditing.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Supported Methods](#supported-methods)
  - [Base Recommenders](#base-recommenders)
  - [Policies](#policies)
  - [Leakage Evaluation](#leakage-evaluation)
  - [Attackers](#attackers)
  - [LLM Audit](#llm-audit)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
  - [Main Bundle](#main-bundle)
  - [Tau Sweep](#tau-sweep)
  - [State-Eta Sweep](#state-eta-sweep)
  - [Top-K Sweep](#top-k-sweep)
  - [Ranker Robustness](#ranker-robustness)
  - [State-Family Sweep](#state-family-sweep)
- [Outputs](#outputs)
  - [Main Output Files](#main-output-files)
  - [Per-Seed Output Files](#per-seed-output-files)
  - [Sweep Output Files](#sweep-output-files)
  - [LLM Audit Output Files](#llm-audit-output-files)
- [Code Organization](#code-organization)
- [Implementation Notes](#implementation-notes)
- [Known Refactor Fixes](#known-refactor-fixes)
- [Troubleshooting](#troubleshooting)
- [Reproducibility Notes](#reproducibility-notes)


---

## Overview

This repository implements a modular pipeline for **privacy-aware recommendation auditing** on **MovieLens-1M**.

The main goal is to measure how much a recommendation policy leaks about a user's latent or short-term state through the recommendation list itself. The codebase supports:

- training base recommenders,
- constructing state-conditioned recommendation settings,
- applying multiple serving / re-ranking policies,
- measuring utility and leakage,
- evaluating certification-style leakage bounds under additive query families,
- testing broader query families,
- training post-hoc attackers,
- and optionally using an LLM as an external auditor.

---

## Features

- Modular Python package layout under `src/`
- Support for multiple recommenders:
  - BPR-MF
  - LightGCN
  - SASRec
- Support for multiple serving policies:
  - State-aware
  - State-independent
  - RA
  - RA+Mix
- Leakage measurement under:
  - certified additive query families
  - full bounded query families
  - diagnostic smooth query families
- Statistical attacker evaluation:
  - Logistic Regression
  - Random Forest
  - GBDT
  - MLP
- Optional LLM-based auditing pipeline
- Seeded multi-run evaluation and aggregation
- Optional sweeps over:
  - `tau`
  - `state_eta`
  - `topk`
  - ranker family
  - state construction family

---

## Repository Structure

```text
ml1m-ra-audit/
├─ README.md
├─ requirements.txt
├─ llm_registry.example.json
├─ run.py
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ utils.py
   ├─ data.py
   ├─ models.py
   ├─ policies.py
   ├─ queries.py
   ├─ attackers.py
   ├─ llm_audit.py
   ├─ reporting.py
   └─ experiment.py
```

### Top-level files

- `run.py`  
  Entry point for the full experiment pipeline.

- `requirements.txt`  
  Python dependency list.

- `llm_registry.example.json`  
  Example registry for configuring LLM audit backends.

### Source modules

- `src/config.py`  
  Global experiment configuration, defaults, and helper utilities such as `get_policy_eta`.

- `src/utils.py`  
  Shared helpers such as seeding, formatting, JSONL writing, hashing, ranking utilities, and small math helpers.

- `src/data.py`  
  MovieLens-1M loading, preprocessing, user-history construction, audit-user filtering, and data splitting.

- `src/models.py`  
  Training and scoring for BPR-MF, LightGCN, and SASRec.

- `src/policies.py`  
  State-aware and privacy-aware re-ranking logic.

- `src/queries.py`  
  Leakage query construction, query-family evaluation, transcript-level statistics, and leakage summaries.

- `src/attackers.py`  
  Statistical attacker datasets, models, and attacker evaluation suites.

- `src/llm_audit.py`  
  Optional LLM-based audit prompts, cache handling, model calls, result normalization, and bootstrap summaries.

- `src/reporting.py`  
  Aggregation, pretty tables, and auxiliary summary exports.

- `src/experiment.py`  
  End-to-end orchestration for single-seed runs, multi-seed bundles, sweeps, candidate selection, and export logic.

---

## Supported Methods

## Base Recommenders

The repository supports the following backbone recommenders:

### 1. BPR-MF
A pairwise ranking matrix factorization baseline trained using sampled triplets.

### 2. LightGCN
A graph-based collaborative filtering model built on user-item interactions.

### 3. SASRec
A sequential recommendation model that uses self-attention over the user's interaction history.

---

## Policies

The following serving policies are supported:

### 1. State-aware
Uses hidden-state-dependent bonuses to adapt the recommendation list to the current user state.

### 2. State-independent
A state-agnostic baseline that ignores the hidden state when serving.

### 3. RA
A privacy-aware re-ranking policy that trades off recommendation utility against leakage-related risk.

### 4. RA+Mix
A randomized mixture between policies, intended to improve privacy certification properties.

---

## Leakage Evaluation

The code supports multiple leakage evaluation families.

### 1. Certified additive query family
Used for exact additive-style certification analysis and separable reranking.

### 2. Full bounded query family
A richer family used for more complete empirical privacy evaluation.

### 3. Diagnostic smooth query family
Useful for debugging and studying non-additive leakage behavior.

Common metrics include:

- recommendation utility:
  - `ndcg`
  - `recall`
- privacy / leakage:
  - `Val.UCB`
  - `Test.qleak`
  - certification rate
- attacker metrics:
  - `Stat.AUC`
- optional LLM metrics:
  - `LLM.AUC`
  - `LLM.SingleAcc`
  - `LLM.PairAcc`

---

## Attackers

Supported statistical attackers include:

- Logistic Regression
- Random Forest
- Gradient Boosted Decision Trees
- MLP

These attackers attempt to infer the user's hidden state from the served recommendation lists.

---

## LLM Audit

The repository optionally supports an LLM-based audit stage.

The LLM auditor can evaluate recommendation lists using:

- single-list inference prompts,
- paired-list comparison prompts,
- scalar/rating-style prompts.

This is disabled by default and intended as an optional auxiliary analysis rather than a required component of the core pipeline.

---

## Requirements

Recommended environment:

- Python `3.10` or `3.11`

Recommended dependencies:

```text
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
requests>=2.31.0
torch>=2.2.0
```

If you use CUDA, install a PyTorch build compatible with your CUDA version.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ml1m-ra-audit
```

### 2. Create a virtual environment

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

This project expects the **MovieLens-1M** dataset directory to contain:

- `ratings.dat`
- `movies.dat`
- `users.dat`

Example layout:

```text
data/ml-1m/
├─ ratings.dat
├─ movies.dat
└─ users.dat
```

### Notes

- The loader expects the original `::`-separated ML-1M files.
- Positive interactions are derived from ratings with `Rating >= 4`.
- Histories are sorted by:
  1. `UserID`
  2. `Timestamp`
  3. `MovieID`

---

## Quick Start

Run the default experiment bundle:

```bash
python run.py --data_dir "/path/to/ml-1m" --out_dir "./outputs"
```

Auto-select device:

```bash
python run.py --data_dir "/path/to/ml-1m" --out_dir "./outputs" --device auto
```

Force CPU:

```bash
python run.py --data_dir "/path/to/ml-1m" --out_dir "./outputs" --device cpu
```

Force CUDA:

```bash
python run.py --data_dir "/path/to/ml-1m" --out_dir "./outputs" --device cuda
```

### Windows example

```powershell
python run.py --data_dir "D:\data\ml-1m" --out_dir "D:\results" --device auto
```

---

## Configuration

Most settings are defined in `src/config.py` through the `Config` dataclass.

Important fields include:

| Field | Description |
|---|---|
| `ranker_name` | Base recommender to use: `bprmf`, `lightgcn`, or `sasrec` |
| `topk` | Number of recommendations served |
| `tau` | Leakage threshold / target used in certification-oriented selection |
| `lambda_grid` | Candidate RA tradeoff values |
| `alpha_grid` | Candidate RA+Mix mixture values |
| `state_family` | Hidden state construction family |
| `state_eta` | State-effect strength used for hidden-state item bonuses |
| `policy_eta` | Override for serving-time eta; if `None`, defaults to `state_eta` |
| `llm_enabled` | Whether to run the optional LLM audit suite |
| `seeds` | Main experiment seeds |
| `sweep_seeds` | Seeds used in sweep runs |
| `run_tau_sweep` | Whether to export tau sweep results |
| `run_eta_sweep` | Whether to export state-eta sweep results |
| `run_k_sweep` | Whether to export top-k sweep results |
| `run_ranker_robustness` | Whether to compare different recommenders |
| `run_state_family_sweep` | Whether to compare hidden-state construction families |

### Important detail

If `policy_eta` is not explicitly set, serving logic should use:

```python
get_policy_eta(cfg)
```

rather than accessing a non-existent `cfg.eta`.

---

## Running Experiments

## Main Bundle

The main bundle runs the core seed loop and exports the main tables.

```bash
python run.py --data_dir "/path/to/ml-1m" --out_dir "./outputs"
```

This typically performs:

1. dataset loading,
2. audit-user construction,
3. train / validation / test user splitting,
4. recommender training,
5. policy serving,
6. leakage evaluation,
7. attacker evaluation,
8. optional LLM audit,
9. aggregation across seeds.

---

## Tau Sweep

To evaluate performance across different certification thresholds, enable `run_tau_sweep=True` in `Config`.

Outputs are written under:

```text
outputs/tau_sweep/
```

---

## State-Eta Sweep

To study the effect of hidden-state strength, enable:

```python
run_eta_sweep = True
```

and set:

```python
eta_grid = (0.02, 0.03, 0.04, 0.06, 0.08)
```

Outputs are written under:

```text
outputs/state_eta_sweep/
```

---

## Top-K Sweep

To evaluate multiple recommendation list sizes, enable:

```python
run_k_sweep = True
```

and configure:

```python
k_grid = (10, 20, 50, 100)
```

Outputs are written under:

```text
outputs/k_sweep/
```

---

## Ranker Robustness

To compare BPR-MF, LightGCN, and SASRec, enable:

```python
run_ranker_robustness = True
```

and configure:

```python
ranker_robustness_models = ("bprmf", "lightgcn", "sasrec")
```

Outputs are written under:

```text
outputs/ranker_robustness/
```

---

## State-Family Sweep

To compare hidden-state construction variants, enable:

```python
run_state_family_sweep = True
```

and configure:

```python
state_family_grid = ("semantic_linear", "sparse_bucket", "history_conditioned")
```

Outputs are written under:

```text
outputs/state_family_sweep/
```

---

## Outputs

## Main Output Files

These are usually written to the main output directory:

- `table1_all_seeds_full.csv`  
  Combined results across all seeds and policies.

- `table1_aggregate_full.csv`  
  Aggregated mean/std summary by policy.

- `table1_pretty_full.csv`  
  Human-readable version of the aggregate table.

- `selection_by_seed.csv`  
  Seed-level summary of selected RA / RA+Mix parameters.

---

## Per-Seed Output Files

Each seed directory may contain files such as:

- `seed_*/table1_seed_fullgamma.csv`
- `seed_*/selection.json`
- `seed_*/fullgamma_candidates.csv`
- `seed_*/fullgamma_mix_candidates.csv`
- `seed_*/top_leaking_cert_queries.csv`
- `seed_*/top_leaking_queries_fullG.csv`
- `seed_*/diagnostic_nonadditive_qleak.csv`
- `seed_*/table2_attacker_auc_suite.csv`
- `seed_*/table6_runtime.csv`

These files are useful for debugging, ablations, and paper-table reconstruction.

---

## Sweep Output Files

Possible sweep outputs include:

- `tau_sweep/tau_sweep_summary.csv`
- `tau_sweep/tau_calibration_summary.csv`
- `state_eta_sweep/state_eta_sweep_summary.csv`
- `k_sweep/k_sweep_summary.csv`
- `ranker_robustness/ranker_robustness_summary.csv`
- `state_family_sweep/state_family_sweep_summary.csv`

---

## LLM Audit Output Files

When `llm_enabled=True`, you may also see:

- `seed_*/llm_audit_summary_policy.csv`
- `seed_*/llm_audit_summary_persona.csv`
- `seed_*/llm_audit_summary_raw.csv`
- `seed_*/llm_audit_raw.jsonl`
- `seed_*/llm_audit_bootstrap_summary.csv`

---

## Code Organization

The project is intentionally split into modules to make the original single-file implementation easier to maintain and extend.

### `src/config.py`
Defines the `Config` dataclass and global policy ordering.

### `src/utils.py`
Contains small utilities reused across the codebase.

### `src/data.py`
Handles MovieLens-1M reading, preprocessing, eligibility filtering, and user splitting.

### `src/models.py`
Contains recommender training and score precomputation.

### `src/policies.py`
Implements serving and re-ranking logic for each policy family.

### `src/queries.py`
Defines leakage queries and evaluation helpers.

### `src/attackers.py`
Builds attacker datasets and runs statistical attacker models.

### `src/llm_audit.py`
Implements optional LLM prompting, inference, and aggregation.

### `src/reporting.py`
Builds aggregate tables and exports summary CSVs.

### `src/experiment.py`
Coordinates the full experiment lifecycle.

---

## Implementation Notes

- Positive interactions are constructed from ratings `>= 4`.
- User histories are chronological.
- Audit users are filtered using state-conditioned constraints.
- Utility is evaluated with ranking metrics such as NDCG and Recall.
- Leakage is measured both by empirical query-family differences and downstream attacker performance.
- The additive certified family is the main object for exact certification-style analysis.
- The broader bounded family is recommended for fuller empirical leakage evaluation.
- LLM evaluation is optional and should be interpreted as an auxiliary audit, not as a replacement for formal or empirical metrics.

---

## Known Refactor Fixes

This multi-file refactor is intended to preserve behavior while fixing several structural issues from the original monolithic version.

### 1. Duplicate config fields removed

The original code defined some fields more than once, including:

- `stateaware_rerank_M`
- `lambda_grid`

The refactor keeps only one canonical definition.

### 2. `cfg.eta` bug fixed

The original code referenced:

```python
cfg.eta
```

but `Config` does not define an `eta` field.

The correct logic is:

```python
get_policy_eta(cfg)
```

which resolves to:

- `cfg.policy_eta` if provided,
- otherwise `cfg.state_eta`.

### 3. Old `run_eta_sweep()` should be retired

The original single-file version contained an older sweep path that attempted to write a non-existent `eta` field. The refactor should keep only:

- `run_state_eta_sweep()`

### 4. Prefer `itertuples()` over `iterrows()`

Where possible, row iteration is converted from:

```python
for _, row in df.iterrows():
```

to:

```python
for row in df.itertuples(index=False):
```

for better speed and fewer dtype surprises.

---

## Troubleshooting

### 1. `FileNotFoundError` for ML-1M files

Check that your dataset directory contains:

- `ratings.dat`
- `movies.dat`
- `users.dat`

and that you passed the correct `--data_dir`.

### 2. CUDA not available

If you requested `--device cuda` but PyTorch cannot detect a GPU, either:

- install a CUDA-enabled PyTorch build, or
- run with:

```bash
--device cpu
```

### 3. LLM audit does not run

Make sure:

- `llm_enabled=True` in `Config`
- your registry file exists
- the API key environment variable is set correctly
- the selected model name exists in the registry

### 4. Output tables are missing

Some outputs appear only when certain options are enabled, such as:

- sweeps,
- LLM audit,
- robustness experiments.

Check the corresponding config flags.

### 5. Very long runtime

Try:

- reducing `epochs`,
- reducing `seeds`,
- setting `minimal_run=True`,
- disabling LLM audit,
- disabling sweeps.

---

## Reproducibility Notes

To improve reproducibility:

- use fixed seeds,
- keep dependency versions pinned,
- log the exact `Config`,
- avoid changing data preprocessing,
- record CUDA / PyTorch versions if using GPU.

Minor numeric variation may still occur across hardware or CUDA backends.
