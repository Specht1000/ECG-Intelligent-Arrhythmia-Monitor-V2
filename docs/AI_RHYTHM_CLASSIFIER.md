# Six-label rhythm classifier

## Scope

This experiment is the first approved V2 complete-exam rhythm benchmark. It is a
multi-label classifier: one 10-second ECG can receive more than one positive
label. It is an engineering benchmark for research only and is not a diagnostic
or clinical-use model.

The six labels are:

| Label | Chapman SNOMED CT | PTB-XL SCP-ECG |
| --- | --- | --- |
| Sinus rhythm | `426783006` | `SR` |
| Sinus bradycardia | `426177001` | `SBRAD` |
| Sinus tachycardia | `427084000` | `STACH` |
| Sinus arrhythmia | `427393009` | `SARRH` |
| Atrial fibrillation | `164889003` | `AFIB` |
| Atrial flutter | `164890007` | `AFLT` |

This initial mapping does not replace the future specialist-approved clinical
taxonomy.

## Data protocol

- Primary data: Chapman-Shaoxing-Ningbo 12-lead ECG database.
- Chapman records selected by the six target codes: 43,366.
- Split: 34,692 training, 4,337 validation, and 4,337 held-out test records.
- The split is reproducible and stratified by label combination.
- External data: all 2,198 PTB-XL records in official fold 10.
- PTB-XL is never used for training, checkpoint selection, or threshold selection.
- Input: 12 standard leads in canonical order, 10 seconds, 100 Hz, physical mV.
- Chapman signals are anti-aliased from 500 Hz to 100 Hz with
  `scipy.signal.resample_poly`.
- Per-lead normalization statistics are calculated from Chapman training data
  only.

Chapman publishes one ECG per subject record and does not expose a separate
patient identifier. PTB-XL official folds respect patient separation.

## Model and training

`RhythmECGNet` is a compact residual one-dimensional CNN with 354,646 trainable
parameters. It produces six independent logits and is trained with weighted
binary cross-entropy. Positive weights compensate for class imbalance and are
capped at 25.

The checkpoint is selected by validation macro average precision. A separate
decision threshold for each label maximizes F1 on Chapman validation data. These
thresholds are then frozen before both evaluations.

Run training from the repository root:

```powershell
.\.venv\Scripts\python.exe host/ai/train_rhythm_classifier.py
```

The complete configuration, normalization values, thresholds, class counts,
software versions, and metrics are saved in
`artifacts/rhythm_classifier/metrics.json`.

## Results

The following results come from the checkpoint trained on 2026-08-31:

| Evaluation set | Macro AP | Macro AUROC |
| --- | ---: | ---: |
| Chapman held-out test | 0.7425 | 0.9625 |
| PTB-XL external fold 10 | 0.4966 | 0.8819 |

Per-class results:

| Label | Chapman support | Chapman AP | Chapman sensitivity | PTB-XL support | PTB-XL AP | PTB-XL sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sinus rhythm | 814 | 0.9363 | 0.9398 | 1,674 | 0.9342 | 0.7772 |
| Sinus bradycardia | 1,655 | 0.9989 | 0.9855 | 64 | 0.1492 | 0.9688 |
| Sinus tachycardia | 726 | 0.9895 | 0.9752 | 82 | 0.8746 | 0.9634 |
| Sinus arrhythmia | 255 | 0.4518 | 0.4510 | 77 | 0.0835 | 0.2208 |
| Atrial fibrillation | 178 | 0.2369 | 0.8146 | 152 | 0.8455 | 0.6776 |
| Atrial flutter | 806 | 0.8417 | 0.9752 | 7 | 0.0924 | 0.8571 |

Average precision must be interpreted together with positive support and
precision. For example, PTB-XL atrial flutter has only seven positive records;
despite high sensitivity, precision is 0.0291, so the estimate is unstable and
not suitable for a performance claim.

The large cross-dataset degradation for several labels indicates acquisition,
population, annotation, or mapping domain shift. In particular, the current
sinus-arrhythmia and atrial-flutter outputs are not robust externally. The model
must not be presented as clinically validated.

## Inference

Run inference on one compatible 12-lead PTB-XL `records100` record:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

The command accepts the record base path, `.hea` path, or `.dat` path and prints
JSON with all six probabilities, thresholds, and independent binary decisions.
It currently requires a 12-lead, 100 Hz, 1,000-sample WFDB record with physical
units in mV and the canonical lead order. Hardware-stream inference requires a
separate adapter after the acquisition contract is confirmed.

## Artifacts

`artifacts/rhythm_classifier/` contains:

- `model.pt`: model weights and preprocessing metadata;
- `metrics.json`: complete reproducible report;
- `evaluation.png`: learning and cross-dataset comparison chart;
- `training_history.csv`: one row per epoch;
- `chapman_split_manifest.csv`: exact primary-data split;
- `chapman_validation_predictions.csv`: validation outputs;
- `chapman_test_predictions.csv`: held-out Chapman outputs;
- `ptbxl_external_predictions.csv`: external PTB-XL outputs.

The dedicated error, subgroup, calibration, and label-interaction analysis is
documented in
[`AI_RHYTHM_ERROR_ANALYSIS.md`](AI_RHYTHM_ERROR_ANALYSIS.md).

## Remaining work

- review the clinical taxonomy with a cardiologist;
- investigate label definitions and domain shift before combined training;
- obtain clinical approval for probability calibration and an abstention policy;
- review the completed error, subgroup, and label-interaction analysis with a
  cardiologist;
- validate on signals produced by the final PCB and acquisition pipeline;
- define specialist adjudication and prospective validation protocols.
