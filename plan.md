# Plan: FAST-NUCES Assignment 2 (Responsible and Explainable AI)

## Objective
Deliver a complete, reproducible, and well-documented implementation of all five assignment parts using the required tools, with executed notebooks, required plots/tables, and a production-style moderation pipeline.

## Deliverables
- part1.ipynb: Baseline DistilBERT classifier
- part2.ipynb: Bias audit across Black vs reference cohorts
- part3.ipynb: Adversarial attack analysis (evasion + poisoning)
- part4.ipynb: Bias mitigation comparison (3 techniques + baseline)
- part5.ipynb: Guardrail pipeline demonstration on 1,000 examples
- pipeline.py: ModerationPipeline class with predict(text)
- requirements.txt: pinned dependencies
- README.md: setup, runtime details, and reproducibility steps
- .gitignore: excludes dataset and checkpoints per instructions

## Non-Negotiable Constraints From Assignment
- Core libraries must be used:
  - transformers
  - torch
  - scikit-learn
  - fairlearn
  - aif360
  - pandas
  - matplotlib
  - seaborn
- Dataset split must be implemented manually with stratified sampling on binarized toxic label.
- Training subset: 100,000 rows.
- Evaluation subset: 20,000 rows.
- Evaluation split must be held out and never used in training.
- DistilBERT model must be used.
- Notebooks must be submitted with outputs already executed.
- Dataset/model binaries must not be committed.

## High-Level Architecture

### Data and split flow
1. Load full CSV.
2. Binarize toxic as label = 1 if toxic >= 0.5 else 0.
3. Perform stratified split to get:
   - train_pool: 100,000 rows
   - eval_pool: 20,000 rows
4. Persist split indices to disk so all parts use the same rows and prevent accidental leakage.

### Model flow
1. Tokenize comment_text with distilbert-base-uncased tokenizer.
2. Fine-tune DistilBERT via HuggingFace Trainer for 3 epochs.
3. Obtain probabilities and predictions on eval subset.
4. Use shared evaluation utilities for metrics and plots.

### Fairness and attacks flow
1. Build subgroup cohorts from eval set using black and white columns.
2. Compute per-cohort and disparity metrics.
3. Run adversarial stress tests on trained model.
4. Compare fairness/accuracy before and after mitigations.

### Guardrail flow
1. Layer 1 regex pre-filter.
2. Layer 2 calibrated model confidence decision.
3. Layer 3 review queue for uncertain cases.

## Recommended Repository Layout
- notebooks/
  - part1.ipynb
  - part2.ipynb
  - part3.ipynb
  - part4.ipynb
  - part5.ipynb
- src/
  - data_utils.py
  - model_utils.py
  - metrics_utils.py
  - fairness_utils.py
  - attack_utils.py
  - calibration_utils.py
  - pipeline.py
- artifacts/
  - split_indices.json
  - part1_checkpoint/
  - part4_best_model/
  - cached_predictions/
- requirements.txt
- README.md
- .gitignore

Note: Assignment asks for root-level files. Final submission should keep required files at root. If using subfolders during development, move/copy final files to root before submission.

## Global Reproducibility Rules
- Fix random seeds for Python, NumPy, Torch, and sampling operations.
- Log versions of Python, CUDA, torch, transformers, sklearn, fairlearn, aif360.
- Save split indices and reuse them in every part.
- Keep a single source of truth for threshold used after Part 1.
- Keep all notebook outputs executed before final push.

## Implementation Steps (Detailed)

## Phase 0: Environment and Project Setup
- [ ] Create virtual environment (or Colab runtime setup).
- [ ] Install pinned dependencies.
- [ ] Verify GPU availability (Colab T4 or local CUDA).
- [ ] Create .gitignore with required patterns:
  - *.csv
  - *.pt
  - *.bin
  - saved_model/
  - artifacts/part1_checkpoint/
  - artifacts/part4_best_model/
- [ ] Initialize repository structure.
- [ ] Create helper modules for reusable logic to avoid copy-paste across notebooks.

## Phase 1: Data Loading, Binarization, and Stratified Split
- [ ] Read jigsaw-unintended-bias-train.csv via pandas.
- [ ] Validate critical columns exist: comment_text, toxic, black, white.
- [ ] Create binary label column:
  - label = (toxic >= 0.5).astype(int)
- [ ] Report class distribution before split.
- [ ] Implement stratified sampling logic with sklearn train_test_split:
  - First isolate 20,000 eval rows stratified on label.
  - Then sample 100,000 training rows from remaining pool stratified on label.
- [ ] Verify no overlap between train and eval indices.
- [ ] Save chosen indices to artifacts/split_indices.json.
- [ ] Print class balance for train and eval to prove stratification preserved.

Acceptance checks
- [ ] Train size exactly 100,000.
- [ ] Eval size exactly 20,000.
- [ ] Label prevalence in train and eval close to full data prevalence.
- [ ] Zero index overlap.

## Phase 2: Part 1 Notebook (Baseline)

### Part 1 required implementation
- [ ] Build part1.ipynb with clear sections:
  1) Setup and imports
  2) Data load and split (reuse saved split)
  3) Tokenization
  4) Training
  5) Evaluation metrics
  6) Curves and threshold analysis
  7) Markdown interpretation
- [ ] Tokenize comment_text with:
  - tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  - truncation=True
  - max_length=128
- [ ] Fine-tune model for 3 epochs using Trainer API.
- [ ] Use standard cross-entropy (default in AutoModelForSequenceClassification).

### Part 1 required outputs
- [ ] Accuracy on eval set.
- [ ] Macro F1.
- [ ] ROC-AUC.
- [ ] Confusion matrix.
- [ ] ROC curve plot.
- [ ] Precision-Recall curve plot.
- [ ] Threshold sweep at 0.3, 0.4, 0.5, 0.6, 0.7 with F1 at each threshold.
- [ ] Markdown justification for chosen operating threshold and platform priorities trade-off.

### Part 1 persistence
- [ ] Save model checkpoint to artifacts/part1_checkpoint.
- [ ] Save selected threshold in a small JSON config for reuse.

Acceptance checks
- [ ] All required metrics shown in output cells.
- [ ] Both ROC and PR curves rendered.
- [ ] Threshold decision explicitly justified.

## Phase 3: Part 2 Notebook (Bias Audit)

### Cohort construction
- [ ] From eval subset, create:
  - high_black: black >= 0.5
  - reference: black < 0.1 and white >= 0.5
- [ ] Print cohort sizes at notebook start.
- [ ] If cohort size is unexpectedly tiny (<50), verify filter logic before continuing.

### Required cohort metrics
- [ ] For each cohort compute:
  - TPR
  - FPR
  - FNR
  - Precision
- [ ] Compute Disparate Impact ratio for over-flagging:
  - FPR(high_black) / FPR(reference)
- [ ] Use AIF360 ClassificationMetric to compute:
  - Statistical parity difference
  - Equal opportunity difference

### Required visualizations
- [ ] Grouped bar chart: TPR, FPR, FNR for both cohorts.
- [ ] Confusion matrix for high_black.
- [ ] Confusion matrix for reference.

### Required analysis
- [ ] Markdown answer identifying largest disparity metric.
- [ ] Explain whether harm is over-flagging (FPR), under-flagging (FNR), or both.
- [ ] Discuss practical consequences for users and platform.

Acceptance checks
- [ ] Summary table includes all required metrics.
- [ ] Both confusion matrices rendered.
- [ ] Disparate impact ratio clearly reported.

## Phase 4: Part 3 Notebook (Adversarial Attacks)

### Attack 1: Character-level evasion
- [ ] Implement perturb(text) with all 3 transformations:
  1) Zero-width space insertion (U+200B)
  2) Unicode homoglyph substitution (a/e/o etc.)
  3) Random character duplication at 20%
- [ ] Keep random seed fixed for repeatability.
- [ ] Select 500 random eval comments meeting:
  - Clean model predicts toxic (label 1)
  - Confidence >= 0.7
- [ ] Apply perturbation and re-score with same model.
- [ ] Compute Attack Success Rate:
  - fraction flipped from predicted toxic to non-toxic
- [ ] Report mean confidence before vs after perturbation.
- [ ] Include compact examples of original vs perturbed text for sanity.

### Attack 2: Label-flipping poisoning
- [ ] Copy 100,000-row training subset.
- [ ] Randomly pick 5% rows (5,000) and flip labels.
- [ ] Retrain fresh DistilBERT from original pre-trained checkpoint.
- [ ] Use same hyperparameters as Part 1.
- [ ] Evaluate on clean eval subset.
- [ ] Report before/after comparison:
  - Accuracy
  - Macro F1
  - FNR (especially emphasized)

### Required analysis
- [ ] Markdown comparison of operational danger under realistic threat models.
- [ ] Discuss attacker capability assumptions for evasion vs poisoning.

Acceptance checks
- [ ] ASR table is shown.
- [ ] Poisoning before/after metrics table is shown.
- [ ] Threat-model reasoning is explicit and grounded.

## Phase 5: Part 4 Notebook (Mitigation)

### Baseline for comparison
- [ ] Bring baseline results from Part 2/Part 1 on same eval cohorts.

### Technique 1: Reweighing (pre-processing)
- [ ] Use AIF360 Reweighing with:
  - privileged: reference cohort condition
  - unprivileged: high_black condition
- [ ] Compute sample weights.
- [ ] Train DistilBERT from scratch with weighted loss.
  - If Trainer cannot directly consume sample_weight, implement custom Trainer.compute_loss to apply per-example weights.
- [ ] Evaluate fairness and accuracy metrics.

### Technique 2: Threshold optimization (post-processing)
- [ ] Use fairlearn ThresholdOptimizer with equalized_odds.
- [ ] Fit using subgroup labels and model scores.
- [ ] Generate adjusted predictions.
- [ ] Sweep fairness tolerance from 0.0 to 0.3.
- [ ] Plot Pareto curve:
  - x-axis: equal opportunity difference
  - y-axis: overall F1

### Technique 3: Oversampling (data-level)
- [ ] In training set, duplicate high_black rows 3 times (4 total including original).
- [ ] Retrain from scratch.
- [ ] Re-evaluate on same eval cohorts.

### Required comparison and analysis
- [ ] Build summary table rows: baseline + 3 techniques.
- [ ] Required columns:
  - Overall F1
  - High-black FPR
  - Reference FPR
  - Statistical parity difference
  - Equal opportunity difference
- [ ] Try to satisfy demographic parity and equalized odds simultaneously.
- [ ] If impossible, compute and present base rates per cohort and explain incompatibility mathematically.

### Persistence
- [ ] Save best-performing mitigated model to artifacts/part4_best_model.

Acceptance checks
- [ ] All three techniques present with proper evaluation.
- [ ] Pareto plot rendered.
- [ ] Incompatibility explanation includes actual base-rate numbers.

## Phase 6: Part 5 Notebook + pipeline.py (Guardrails)

### pipeline.py requirements
- [ ] Implement ModerationPipeline class.
- [ ] Implement predict(text) returning structured decision dictionary.

### Layer 1: Regex input filter
- [ ] Create BLOCKLIST dictionary keyed by categories:
  - direct_threat (>=5 patterns)
  - self_harm_directed (>=4 patterns)
  - doxxing_stalking (>=4 patterns)
  - dehumanization (>=4 patterns)
  - coordinated_harassment (>=3 patterns)
- [ ] Compile with re.IGNORECASE.
- [ ] Use word boundaries where required.
- [ ] Include required regex features:
  - At least one capturing group for threat verb alternatives.
  - Non-capturing group (?:human|people|person) in dehumanization category.
  - At least one lookahead (?=...) in coordinated_harassment.
- [ ] Input filter returns category-aware decision.

### Layer 2: Calibrated model
- [ ] Load best mitigated model.
- [ ] Produce raw probabilities.
- [ ] Fit CalibratedClassifierCV(method="isotonic") on calibration split or held-out fold from training data.
- [ ] Inference decision rules:
  - prob >= 0.6 => block
  - prob <= 0.4 => allow
  - 0.4 < prob < 0.6 => route to review

### Layer 3: Review queue
- [ ] Return review decision for uncertain predictions.
- [ ] Ensure structured response includes decision, layer, confidence.

### part5.ipynb demonstration
- [ ] Sample 1,000 comments from eval subset.
- [ ] Run pipeline on each comment.
- [ ] Report layer distribution by count and fraction.
- [ ] Plot pie or bar chart of layer distribution.
- [ ] For auto-action subset (non-review model decisions), report:
  - F1
  - Precision
  - Recall
- [ ] For review subset, report actual toxic/non-toxic breakdown.
- [ ] Perform threshold-band sensitivity:
  - baseline 0.4 to 0.6
  - narrow 0.45 to 0.55
  - wide 0.3 to 0.7
- [ ] Compare review volume and auto-action quality across bands.
- [ ] Provide final threshold recommendation in markdown.

Acceptance checks
- [ ] pipeline.py runs without errors.
- [ ] predict(text) returns valid dict for all paths.
- [ ] Layer-distribution plot rendered.
- [ ] Band sensitivity results shown with numeric comparison.

## Phase 7: requirements.txt, README.md, and Final Submission Hardening

### requirements.txt
- [ ] Pin all required package versions.
- [ ] Include notebook/runtime dependencies only as needed.

Suggested baseline pins (adjust to tested environment)
- python==3.10.x (documented in README)
- torch==2.2.2
- transformers==4.41.2
- datasets==2.19.1
- scikit-learn==1.4.2
- fairlearn==0.10.0
- aif360==0.6.1
- pandas==2.2.2
- numpy==1.26.4
- matplotlib==3.8.4
- seaborn==0.13.2
- scipy==1.13.1
- jupyter==1.0.0

### README.md
- [ ] Document Python version.
- [ ] Document GPU used (example: Colab T4).
- [ ] Explain data placement and expected filename.
- [ ] Add step-by-step reproduction commands.
- [ ] Explain notebook execution order (Part 1 to Part 5).
- [ ] Mention that split indices are reused across notebooks.

### Submission hardening checklist
- [ ] Ensure root contains required files exactly as instructed.
- [ ] Ensure all notebook outputs are visible.
- [ ] Ensure .gitignore excludes forbidden large files.
- [ ] Ensure no CSV/checkpoint accidentally tracked.
- [ ] Ensure commit history is incremental, not single commit.

## Manual QA and Verification Checklist

### Data integrity
- [ ] Confirm split sizes and no overlap.
- [ ] Confirm label binarization threshold applied correctly.
- [ ] Confirm stratification preserved class ratio.

### Model integrity
- [ ] Confirm DistilBERT checkpoint is distilbert-base-uncased.
- [ ] Confirm max_length=128 and truncation=True in tokenization.
- [ ] Confirm exactly 3 epochs baseline training.

### Metric integrity
- [ ] Independently recompute key metrics from predictions using sklearn.
- [ ] Validate confusion matrix totals match sample counts.
- [ ] Validate cohort-specific metrics use correct cohort masks.

### Fairness integrity
- [ ] Confirm cohort definitions exactly match thresholds.
- [ ] Confirm disparate impact uses FPR ratio in required direction.
- [ ] Confirm AIF360 metrics computed with correct privileged/unprivileged setup.

### Attack integrity
- [ ] Verify perturb(text) applies all 3 modifications.
- [ ] Confirm ASR denominator uses only selected 500 toxic confident predictions.
- [ ] Confirm poisoning retraining starts from original pre-trained model, not fine-tuned model.

### Pipeline integrity
- [ ] Verify regex categories and minimum pattern counts.
- [ ] Verify required regex constructs exist (capturing group, non-capturing group, lookahead).
- [ ] Verify calibrated confidence thresholds route decisions correctly.
- [ ] Verify review queue receives only uncertain range.

### Final packaging integrity
- [ ] Open each notebook and spot-check that outputs are present.
- [ ] Run quick import test for pipeline.py.
- [ ] Validate required root-level filenames.
- [ ] Verify git status clean of disallowed large artifacts.

## Risk Register and Mitigations

Risk: Data leakage between train and eval.
Mitigation: Save and reuse split indices, assert no overlap each notebook run.

Risk: Cohorts too small for stable fairness metrics.
Mitigation: Print cohort counts, include interpretation caveats, avoid overclaiming from tiny cohorts.

Risk: Trainer weighting support mismatch for reweighing.
Mitigation: Use custom Trainer compute_loss to multiply per-example loss by sample weights.

Risk: Calibration overfitting.
Mitigation: Fit calibrator on separate calibration fold from train subset, not eval subset.

Risk: Runtime limits on free Colab.
Mitigation: Cache tokenized datasets and predictions; checkpoint models; avoid retraining unless needed.

Risk: Missing executed outputs at submission.
Mitigation: Final execution sweep and explicit notebook output verification checklist.

## Commit Plan (To Protect Grading)
- [ ] Commit 1: repo scaffold + setup files
- [ ] Commit 2: part1 baseline complete
- [ ] Commit 3: part2 bias audit complete
- [ ] Commit 4: part3 attacks complete
- [ ] Commit 5: part4 mitigation complete
- [ ] Commit 6: pipeline.py + part5 complete
- [ ] Commit 7: final QA pass + README polish

Suggested commit message style
- feat(part1): baseline distilbert training and threshold analysis
- feat(part2): cohort bias audit with fairness metrics
- feat(part3): evasion and poisoning attacks with impact analysis
- feat(part4): mitigation techniques and Pareto tradeoff
- feat(part5): moderation pipeline with layered guardrails
- chore(submission): finalize executed notebooks and docs

## Definition of Done (Assignment Complete)
- [ ] All required files exist at repository root.
- [ ] All notebooks execute end-to-end without errors.
- [ ] All notebook outputs are visible and include required plots/tables.
- [ ] pipeline.py exposes working ModerationPipeline.predict(text).
- [ ] requirements.txt is pinned and tested.
- [ ] README includes exact reproduction details.
- [ ] .gitignore prevents disallowed large file commits.
- [ ] Git history is incremental and meaningful.
