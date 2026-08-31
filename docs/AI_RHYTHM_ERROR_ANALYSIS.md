# Rhythm classifier error analysis

## Purpose

This analysis examines the first six-label rhythm classifier beyond aggregate
scores. It covers per-class errors, demographic and signal-quality subgroups,
probability calibration, threshold proximity, label interactions, and the domain
shift from Chapman to PTB-XL.

The analysis is retrospective and exploratory. It does not define a clinical
uncertainty threshold, abstention policy, or diagnostic workflow.

Run it from the repository root after training the rhythm classifier:

```powershell
.\.venv\Scripts\python.exe host/ai/analyze_rhythm_errors.py
```

## Protocol

- Calibration temperatures are fitted separately for each label using Chapman
  validation predictions only.
- Raw and temperature-scaled calibration are evaluated on the Chapman held-out
  test and PTB-XL external fold 10.
- Classification decisions remain those made with the original validation-set
  thresholds. Calibration does not silently change the reported classifier.
- Subgroup metrics cover sex and age in both datasets and the published
  signal-quality annotations in PTB-XL.
- Threshold proximity is descriptive. No margin is selected as an approved
  review or abstention region.

## Main findings

The analysis contains 39,210 binary label decisions from 6,535 ECGs. There are
2,782 false-positive or false-negative outcomes.

### Domain shift

| Label | Chapman AP | PTB-XL AP | Change |
| --- | ---: | ---: | ---: |
| Sinus rhythm | 0.9363 | 0.9342 | -0.0021 |
| Sinus bradycardia | 0.9989 | 0.1492 | -0.8497 |
| Sinus tachycardia | 0.9895 | 0.8746 | -0.1149 |
| Sinus arrhythmia | 0.4518 | 0.0835 | -0.3682 |
| Atrial fibrillation | 0.2369 | 0.8455 | +0.6086 |
| Atrial flutter | 0.8417 | 0.0924 | -0.7493 |

The prevalence changes are also large. Sinus rhythm represents 18.8% of the
selected Chapman test labels but 76.2% in PTB-XL fold 10. Atrial flutter changes
from 18.6% to 0.3%. Therefore, the AP differences do not measure morphology shift
alone; prevalence and annotation policy are important contributors.

### Fibrillation and flutter interaction

The model does not reliably separate atrial fibrillation from atrial flutter:

- all 178 Chapman atrial-fibrillation records also cross the atrial-flutter
  threshold;
- 64.0% of the 806 Chapman atrial-flutter records cross the
  atrial-fibrillation threshold;
- the same interaction appears externally, but PTB-XL fold 10 contains only
  seven atrial-flutter positives.

The reference labels do not explain this result through ordinary co-labeling:
Chapman has zero AFIB/AFLT co-occurrences in the held-out split. This points to a
model discrimination or cross-dataset labeling problem that requires waveform
review and specialist input. The software does not impose mutual exclusivity,
because the approved task is multi-label and such a rule has not been clinically
approved.

### Calibration

Temperature scaling improves expected calibration error in only 6 of the 12
dataset-class evaluations. It helps PTB-XL sinus rhythm and sinus arrhythmia but
worsens PTB-XL atrial fibrillation and atrial flutter. The result confirms that a
calibrator fitted on Chapman cannot generally correct PTB-XL domain shift.

The largest raw PTB-XL calibration error is sinus arrhythmia at 0.5626; scaling
reduces it to 0.4961, which remains unacceptable for a calibrated probability
claim.

### Subgroups

Sinus arrhythmia is the weakest repeated subgroup result. With at least ten
positives:

- PTB-XL age 60-79: sensitivity 0.000 across 18 positives;
- PTB-XL age 40-59: sensitivity 0.0526 across 19 positives;
- Chapman age 60-79: sensitivity 0.0612 across 49 positives;
- Chapman age 40-59: sensitivity 0.1296 across 54 positives.

These are retrospective observations, not fairness or clinical-validation
claims. Small supports and dataset composition must be considered.

### Threshold proximity

A 0.05 probability margin around the threshold contains 65.8% of Chapman atrial
fibrillation errors while covering 13.2% of its decisions. It also contains about
half of the Chapman sinus-arrhythmia errors while covering 7.8% of decisions.
However, the same margin captures only 1.3% of PTB-XL sinus-bradycardia errors.
Consequently, one universal margin is not justified and no abstention rule has
been adopted.

## Experimental explanation view

Integrated gradients can be generated for one compatible record and class:

```powershell
.\.venv\Scripts\python.exe host/ai/explain_rhythm_prediction.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --class-name sinus_rhythm
```

The baseline is the zero-normalized input, corresponding to the per-lead Chapman
training mean. Red ECG samples have greater absolute attribution for the selected
logit. The accompanying JSON reports relative attribution by lead.

Integrated gradients measures sensitivity of this model to its input. It does
not show medical causality, prove that the highlighted morphology is clinically
relevant, or prove that the prediction is correct. The final explanation method
and presentation still require cardiologist review.

## Artifacts

`artifacts/rhythm_error_analysis/` contains:

- `report.json`: concise machine-readable findings;
- `summary.png`: visual summary;
- `all_class_outcomes.csv`: every binary label outcome;
- `all_errors.csv`: all false-positive and false-negative outcomes;
- `highest_confidence_errors.csv`: up to 25 strong errors per class and type;
- `calibration_metrics.csv` and `calibration_bins.csv`;
- `subgroup_metrics.csv`;
- `threshold_proximity.csv`;
- `domain_shift.csv`;
- `label_interactions.csv`;
- `true_label_cooccurrence.csv`.

`artifacts/rhythm_explanations/` contains generated explanation plots and their
JSON metadata.

## Required next decisions

Software analysis alone cannot settle the following items:

- whether AFIB and AFLT must be mutually exclusive in the final clinical task;
- whether the Chapman and PTB-XL rhythm definitions are sufficiently equivalent
  for combined training;
- which error costs, probability calibration target, and review policy are
  acceptable to the cardiologist;
- which explanation format is useful and safe in the final interface.
