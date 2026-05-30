# Model B Independent Binary Rhythm Pipeline

This folder contains an independent, reproducible pipeline for the binary task:

- Negative class: `normal_sinus`
- Positive class: `arrhythmia_or_abnormal`
- Target: `rhythm_binary`
- Split: 80/20 by `case_id`, with no overlap between train and test

The pipeline does not use raw ECG windows, multiclass labels, administrative/intraoperative predictors, or the previous broad binary feature set.

## Required Base Dataset

The pipeline expects:

```bash
data/processed/binary_rhythm_modeling_dataset.parquet
```

If this file is missing, the Model B scripts fail with a clear message. They do not rebuild legacy ECG or upstream tabular flows automatically.

## Model B Variables

Exactly these 25 original variables are used:

```python
[
    "rr_prev",
    "rr_next",
    "hr_inst_from_rr_prev",
    "position_in_case",
    "rr_prev_rolling_mean_5",
    "rr_prev_rolling_std_5",
    "rr_prev_rolling_mean_20",
    "rr_prev_rolling_std_20",
    "rr_rmssd_5",
    "rr_rmssd_20",
    "rr_pnn50_5",
    "rr_pnn50_20",
    "local_hr_mean_5",
    "local_hr_mean_20",
    "age",
    "sex",
    "bmi",
    "asa",
    "preop_htn",
    "preop_dm",
    "preop_hb",
    "preop_na",
    "preop_k",
    "preop_gluc",
    "preop_cr",
]
```

Forbidden columns such as `rhythm_label`, `rhythm_binary`, `beat_type`, `case_id`, identifiers, outcomes, procedure text, and intraoperative/administrative variables are not predictors. `case_id` is used only for grouped splitting.

## Step-by-Step Run

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

The training script performs the test evaluation once after hyperparameter search. `evaluate_model_b.py` then reads saved artifacts and writes the Markdown reports.

## Full Run

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5
python model_b_pipeline/evaluate_model_b.py
```

Optional random forest:

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5 --include-random-forest
```

## Debug Mode

`--debug` sets:

- `n_iter = 3`
- `n_splits = 2`
- `max_cases = 80`, unless a different `--max-cases` is provided

This mode is intended to verify plumbing and output generation quickly. Do not interpret debug metrics as final model performance.

## Outputs

Reports and tables:

```bash
reports/model_b/
reports/model_b/tables/
reports/model_b/figures/
```

Model artifacts:

```bash
models/model_b/model_b_best_pipeline.joblib
models/model_b/model_b_feature_columns.json
models/model_b/model_b_metadata.json
```

The reduced parquet and model artifacts are local generated outputs. The repository `.gitignore` excludes processed datasets and `models/model_b/`, so they are not intended to be versioned.

## Metrics

The main selection metric is CV `balanced_accuracy` on train folds grouped by `case_id`.

The final test report includes:

- balanced accuracy
- accuracy
- precision for `arrhythmia_or_abnormal`
- recall/sensitivity for `arrhythmia_or_abnormal`
- specificity for `normal_sinus`
- F1 for `arrhythmia_or_abnormal`
- ROC-AUC and Average Precision when scores are available
- absolute and normalized confusion matrices

For scored models, thresholds are selected on train only:

- default score threshold
- Youden J
- max F1 for the abnormal class

The persisted best pipeline is selected by highest mean CV balanced accuracy, not by test performance.

## Tests

```bash
pytest model_b_pipeline/tests_model_b.py
```

The tests cover feature safety, split behavior, preprocessing, metrics, and debug output generation on synthetic data.

## Clinical Use Warning

This is a research pipeline for model development and auditing. It is not validated for clinical decision-making and must not be used as a clinical device or diagnostic tool.
