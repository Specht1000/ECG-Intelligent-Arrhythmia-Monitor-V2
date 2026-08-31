# Experimental ECG Anomaly Baseline

## Scope

This experiment is an initial engineering benchmark for binary, complete-exam
classification. It estimates whether a PTB-XL ECG is normal or contains at least
one annotated diagnostic abnormality. It does not define the final V2 arrhythmia
taxonomy, is not a clinical diagnostic model, and must not be used for patient
care.

## Labels

The target is derived only from SCP-ECG statements marked as diagnostic in
`scp_statements.csv`:

- **Normal (0):** `NORM` is the only diagnostic superclass.
- **Abnormal (1):** at least one diagnostic superclass is `MI`, `STTC`, `CD`, or
  `HYP`.
- **Excluded:** no supported binary target can be derived from the diagnostic
  statements.

The abnormal target therefore includes myocardial infarction, ST/T changes,
conduction disturbances, and hypertrophy. It is broader than arrhythmia and must
be described as an anomaly benchmark.

## Data protocol

- Dataset: [PTB-XL 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
- Input: all 12 leads, 10 seconds, 100 Hz (`records100`).
- Training: official stratified folds 1 through 8.
- Validation: fold 9.
- Test: fold 10.
- Leakage control: PTB-XL patient groups do not cross these folds, as verified by
  the repository dataset audit.
- Normalization: per-lead mean and standard deviation computed from training data
  only.
- Decision threshold: chosen on validation data by maximum balanced accuracy,
  then frozen for the test evaluation.

The 100 Hz records reduce CPU training cost for this baseline. Later experiments
must evaluate whether the intended 500 Hz acquisition provides a material benefit.

## Model and outputs

`AnomalyECGNet` is a compact one-dimensional residual convolutional network. The
training script reports AUROC, average precision, sensitivity, specificity,
balanced accuracy, F1 score, Brier score, and a simple calibration error estimate.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe host/ai/train_anomaly_baseline.py
```

Outputs are written to `artifacts/anomaly_baseline/`:

- `metrics.json`: configuration, label definition, split counts, and metrics;
- `training_history.csv`: loss and validation AUROC by epoch;
- `validation_predictions.csv` and `test_predictions.csv`: probabilities;
- `evaluation.png`: learning curve, ROC, precision-recall, and confusion matrix;
- `model.pt`: model weights and inference metadata.

Run inference on one PTB-XL low-resolution record:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_anomaly.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

The inference command accepts a WFDB base path, `.hea` path, or `.dat` path. It
prints the experimental abnormal probability, validation-selected threshold, and
binary prediction as JSON. It currently accepts only the validated PTB-XL 100 Hz,
10-second, 12-lead format; future hardware inference requires an explicitly
approved preprocessing and signal-quality pipeline.

## Held-out error analysis

Run the reproducible error analysis after training:

```powershell
.\.venv\Scripts\python.exe host/ai/analyze_anomaly_errors.py
```

The analysis reports false negatives by diagnostic superclass and SCP-ECG code,
descriptive performance by sex, age group, and signal-quality annotation, a
reliability diagram, validation-fitted temperature scaling, and an exploratory
uncertainty region selected on validation data. It also creates a PDF containing
the complete 12-lead waveforms of the most confident false negatives.

Subgroup comparisons are descriptive and may be confounded by different diagnosis
prevalence. The uncertainty region is not an approved abstention policy.

## Limitations

- The labels are database annotations, not new specialist adjudications.
- This task detects broad diagnostic abnormalities, not all possible arrhythmias.
- Performance on PTB-XL does not establish performance on the Chapman dataset,
  the future PCB, or real patients.
- Cross-dataset evaluation, probability calibration, signal-quality rejection,
  uncertainty handling, subgroup analysis, and specialist validation remain
  necessary.
