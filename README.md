# FAST-NUCES Assignment 2: Responsible and Explainable AI

This repository contains a complete implementation plan and code scaffold for all required assignment parts:

- `part1.ipynb`: baseline DistilBERT model
- `part2.ipynb`: bias audit
- `part3.ipynb`: adversarial attacks
- `part4.ipynb`: mitigation techniques
- `part5.ipynb`: guardrail pipeline demo
- `pipeline.py`: `ModerationPipeline` class

## Environment

- Python: 3.10.x
- Recommended runtime: Google Colab T4 GPU (free tier), or local CUDA GPU

## Required Libraries

Install pinned dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Expected file at repository root:

- `train.csv`

The code automatically infers toxicity score column from one of:

- `toxic`
- `target`
- `toxicity`

and standardizes it to `toxic_score`, with binary label:

- `label = 1` if score `>= 0.5`
- `label = 0` otherwise

## Execution Order

Run notebooks in this sequence:

1. `part1.ipynb`
2. `part2.ipynb`
3. `part3.ipynb`
4. `part4.ipynb`
5. `part5.ipynb`

Why this order:

- Part 1 creates split indices and baseline checkpoint.
- Parts 2-4 reuse those fixed splits/checkpoints to avoid leakage.
- Part 4 saves best mitigated model used by Part 5.

## Reproducibility

- Fixed random seed (`42`) is used in split creation and sampling.
- Split indices are stored in `artifacts/splits/split_indices.json` and reused across parts.
- Threshold selection from Part 1 is cached in `artifacts/cache/part1_threshold_config.json`.

## Important Submission Rules

- Keep notebook outputs executed before submission.
- Do not commit dataset or model binaries.
- `.gitignore` already excludes `.csv`, `.pt`, `.bin`, and model artifact folders.
- Use incremental commits (single-commit history is penalized).

## Project Structure

- `src/`: reusable helper modules for data, model, metrics, fairness, attacks, mitigation, calibration
- `artifacts/`: split metadata, caches, and local model outputs
- `plan.md`: full implementation plan and quality checklist
