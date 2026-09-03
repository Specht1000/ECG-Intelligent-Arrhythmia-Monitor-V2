# Bipolar Limb-Lead Rhythm Classifier

## Status and intended use

This is an engineering research benchmark for the four-electrode ECG scope. It is
not a medical device, a diagnostic model, or evidence of clinical performance.

The model uses only the three standard bipolar limb leads:

- Lead I: `LA - RA`
- Lead II: `LL - RA`
- Lead III: `LL - LA = II - I`

The four physical electrodes are RA, LA, LL, and RL. RL is the reference/return
electrode and is not an independent model channel. Lead III is mathematically
dependent on leads I and II, so the model receives three waveforms but only two
independent voltage differences.

Augmented limb leads (`aVR`, `aVL`, and `aVF`) and precordial leads (`V1` through
`V6`) are outside the current project scope.

## Training design

The classifier is a compact one-dimensional convolutional neural network with
351,406 trainable parameters. Its input is a 10-second, 100 Hz segment with the
channel order `I`, `II`, `III`. The output is a multi-label probability vector for:

- sinus rhythm;
- sinus bradycardia;
- sinus tachycardia;
- sinus arrhythmia;
- atrial fibrillation;
- atrial flutter.

Chapman-Shaoxing-Ningbo is the primary dataset. All 45,150 valid records are
included, including records without one of the six target labels as all-negative
background examples. The split is stratified by label combination: 80% training,
10% validation, and 10% held-out test. The validation set selects one decision
threshold per class by maximum F1. PTB-XL official fold 10 remains fully external
and is never used for training or threshold selection.

Training used four epochs, a batch size of 256, AdamW with a learning rate of
0.001, weighted binary cross-entropy, and seed `20260831`. The best checkpoint was
the fourth epoch, with validation macro average precision of 0.7196.

## Dataset-level results

| Evaluation set | Records | Macro AP | Macro AUROC | Macro F1 | Micro F1 | Exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chapman held-out test | 4,515 | 0.7185 | 0.9590 | 0.7178 | 0.8516 | 0.7417 |
| PTB-XL external fold 10 | 2,198 | 0.4702 | 0.8657 | 0.4499 | 0.6951 | 0.6069 |

The difference between the held-out and external results is evidence of dataset
shift. The PTB-XL result is the more relevant warning about generalization, but it
is still not a hardware-domain or clinical validation result.

## Per-class held-out results

### Chapman held-out test

| Class | Support | AP | AUROC | Sensitivity | Specificity | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sinus rhythm | 814 | 0.9291 | 0.9843 | 0.9373 | 0.9568 | 0.8267 | 0.8785 |
| Sinus bradycardia | 1,657 | 0.9976 | 0.9987 | 0.9861 | 0.9843 | 0.9732 | 0.9796 |
| Sinus tachycardia | 724 | 0.9897 | 0.9980 | 0.9613 | 0.9908 | 0.9521 | 0.9567 |
| Sinus arrhythmia | 257 | 0.3443 | 0.8749 | 0.2840 | 0.9735 | 0.3925 | 0.3296 |
| Atrial fibrillation | 178 | 0.2444 | 0.9278 | 0.6404 | 0.9048 | 0.2163 | 0.3234 |
| Atrial flutter | 804 | 0.8057 | 0.9702 | 0.9540 | 0.9307 | 0.7490 | 0.8392 |

### PTB-XL external fold 10

| Class | Support | AP | AUROC | Sensitivity | Specificity | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sinus rhythm | 1,674 | 0.9256 | 0.8058 | 0.7754 | 0.7653 | 0.9134 | 0.8388 |
| Sinus bradycardia | 64 | 0.1364 | 0.9112 | 0.9844 | 0.8360 | 0.1525 | 0.2642 |
| Sinus tachycardia | 82 | 0.8484 | 0.9934 | 0.9390 | 0.9830 | 0.6814 | 0.7897 |
| Sinus arrhythmia | 77 | 0.0729 | 0.6399 | 0.1299 | 0.9505 | 0.0870 | 0.1042 |
| Atrial fibrillation | 152 | 0.8035 | 0.9747 | 0.5132 | 0.9941 | 0.8667 | 0.6446 |
| Atrial flutter | 7 | 0.0340 | 0.8693 | 0.8571 | 0.9115 | 0.0300 | 0.0580 |

AP is average precision. Unlike AUROC, it exposes poor precision more clearly for
rare labels. The external atrial-flutter estimate is especially unstable because
only seven positive records are present.

## Error analysis

The analysis covers 40,278 binary class outcomes and 2,754 false-positive or
false-negative outcomes. The principal findings are:

- PTB-XL atrial flutter has the weakest external AP at 0.0340.
- Sinus bradycardia has the largest Chapman-to-PTB-XL AP decrease, from 0.9976 to
  0.1364.
- Of 147 PTB-XL exams without any target label, 87.8% receive at least one false
  positive prediction.
- On Chapman, atrial fibrillation is predicted on 49.9% of flutter records, while
  atrial flutter is predicted on 99.4% of fibrillation records. The current model
  therefore does not reliably separate these rhythms.
- Per-class temperature scaling improves expected calibration error in only two
  of twelve class/dataset evaluations. The output values must not be interpreted
  as clinically calibrated probabilities.

## Indicative comparison with the historical 12-lead experiment

| Evaluation | Historical 12-lead | Current bipolar | Change |
| --- | ---: | ---: | ---: |
| Chapman macro AP | 0.7425 | 0.7185 | -0.0240 |
| Chapman macro AUROC | 0.9625 | 0.9590 | -0.0035 |
| PTB-XL macro AP | 0.4966 | 0.4702 | -0.0264 |
| PTB-XL macro AUROC | 0.8819 | 0.8657 | -0.0161 |
| PTB-XL exact match | 0.5742 | 0.6069 | +0.0328 |

This is not a controlled ablation. The historical experiment excluded Chapman
records without target labels, while the bipolar experiment retains them as
background negatives. Therefore, the differences combine the lead reduction with
a training-policy improvement and must not be attributed only to the number of
leads.

## Reproduction

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe host/ai/train_rhythm_classifier.py
.\.venv\Scripts\python.exe host/ai/analyze_rhythm_errors.py
```

The main outputs are stored in:

- `artifacts/bipolar_rhythm_classifier/model.pt`
- `artifacts/bipolar_rhythm_classifier/metrics.json`
- `artifacts/bipolar_rhythm_classifier/evaluation.png`
- `artifacts/bipolar_rhythm_error_analysis/report.json`
- `artifacts/bipolar_rhythm_error_analysis/summary.png`

## Required next validation

Before this work can support any clinical claim, it needs acquisition from the
actual analog front end, channel-order and polarity verification, signal-quality
gating, patient-level external cohorts, cardiologist-adjudicated labels,
calibration and abstention studies, and prospective clinical validation. The
current checkpoint is suitable only as a reproducible PFE research baseline.
