# ECG V2 — PFE

Software foundation for a four-electrode bipolar limb-lead ECG using an STM32
Nucleo, a dedicated acquisition PCB, and AI-assisted analysis on a computer.

Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) first. It contains the current
decisions and the pending items that must not be resolved by assumption.

## Current status

- V2 context separated from the legacy V1 documentation;
- functional architecture and responsibility boundaries documented;
- bipolar I, II, and III handling implemented on the host, with III derived from
  synchronized I and II samples;
- the earlier 12-lead reconstruction retained only for experiment
  reproducibility;
- unit tests for Einthoven's identity and canonical bipolar-lead order;
- initial logical STM32-to-PC contract documented without fixing the physical
  transport.
- reproducible structural and label audits for Chapman-Shaoxing-Ningbo and
  PTB-XL, with machine-readable reports under `artifacts/dataset_audits/`.
- an experimental PTB-XL normal-versus-abnormal baseline with official,
  patient-separated folds and reproducible evaluation artifacts.
- an experimental six-label, multi-label bipolar rhythm classifier trained on
  Chapman I/II/III and evaluated on held-out Chapman data and PTB-XL fold 10.
- an advanced multi-dataset bipolar rhythm model with calibrated probabilities,
  constrained thresholds, quality/uncertainty abstention, error analysis, and
  controlled lead-count and sampling-rate ablations.

## Repository structure

```text
docs/                   documentation and references
firmware/               future STM32 Nucleo application
host/ecg_v2/            host-side processing library
host/tests/             unit tests
PROJECT_CONTEXT.md      source of truth for project decisions
```

## Run the tests

With Python 3.7 or newer, from the repository root:

```powershell
python -m unittest discover -s host/tests -v
```

Run a numerical reconstruction demonstration:

```powershell
python host/demo_leads.py
```

Plot and save a random source-dataset ECG from Chapman or PTB-XL:

```powershell
python host/ai/plot_random_ecg.py
```

Open the generated PNG and make the random selection reproducible:

```powershell
python host/ai/plot_random_ecg.py --dataset ptbxl --seed 42 --show
```

The lead-reconstruction demonstration has no external dependencies. Dataset tools
and model training use the machine-learning environment below.

## Train the experimental anomaly baseline

This first binary benchmark distinguishes NORM-only PTB-XL exams from exams with
at least one abnormal diagnostic superclass. It is not the final arrhythmia
taxonomy and is not a clinical diagnostic model.

```powershell
.\.venv\Scripts\python.exe host/ai/train_anomaly_baseline.py
```

The script uses PTB-XL folds 1-8 for training, fold 9 for validation and threshold
selection, and fold 10 exactly once for testing. See
[`docs/AI_ANOMALY_BASELINE.md`](docs/AI_ANOMALY_BASELINE.md) for the precise label
and evaluation protocol.

Run experimental inference on one compatible PTB-XL record:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_anomaly.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

Analyze held-out errors, subgroups, calibration, and an exploratory uncertainty
region:

```powershell
.\.venv\Scripts\python.exe host/ai/analyze_anomaly_errors.py
```

## Train the bipolar six-label rhythm classifier

The first approved V2 rhythm benchmark covers sinus rhythm, sinus bradycardia,
sinus tachycardia, sinus arrhythmia, atrial fibrillation, and atrial flutter. It
trains from only bipolar leads I, II, and III in Chapman and uses the same three
channels from PTB-XL fold 10 as a strictly external evaluation set.

```powershell
.\.venv\Scripts\python.exe host/ai/train_rhythm_classifier.py
```

Run inference on one compatible PTB-XL `records100` record. The loader reads the
source record but supplies only I, II, and III to the model:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

See [`docs/AI_BIPOLAR_RHYTHM_CLASSIFIER.md`](docs/AI_BIPOLAR_RHYTHM_CLASSIFIER.md)
for the current input definition, evaluation protocol, results, and limitations.
The earlier 12-lead result remains documented in
[`docs/AI_RHYTHM_CLASSIFIER.md`](docs/AI_RHYTHM_CLASSIFIER.md) as a historical
comparison only.

Analyze per-class errors, calibration, demographic subgroups, threshold
proximity, and Chapman-to-PTB-XL domain shift:

```powershell
.\.venv\Scripts\python.exe host/ai/analyze_rhythm_errors.py
```

Create an experimental integrated-gradients explanation for one class:

```powershell
.\.venv\Scripts\python.exe host/ai/explain_rhythm_prediction.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --class-name sinus_rhythm
```

The current bipolar findings and interpretation limits are included in
[`docs/AI_BIPOLAR_RHYTHM_CLASSIFIER.md`](docs/AI_BIPOLAR_RHYTHM_CLASSIFIER.md).
The earlier 12-lead error analysis remains in
[`docs/AI_RHYTHM_ERROR_ANALYSIS.md`](docs/AI_RHYTHM_ERROR_ANALYSIS.md) for
historical comparison only.

## Train the advanced multi-dataset rhythm model

The advanced research pipeline combines Chapman training records with PTB-XL
folds 1-8, validates on Chapman validation records plus PTB-XL fold 9, and keeps
the Chapman held-out split and PTB-XL fold 10 as separate tests. It adds
asymmetric loss, hard-negative handling, atrial hierarchy, deterministic rhythm
features, masked-signal pretraining, 12-lead teacher distillation, probability
calibration, constrained thresholds, and an experimental abstention gate.

```powershell
.\.venv\Scripts\python.exe host/ai/train_advanced_rhythm_classifier.py
```

Run inference from either a compatible WFDB record or a NumPy array in mV. A
NumPy input may contain I/II/III or only independent leads I/II; in the latter
case, lead III is reconstructed as `II - I` and reported in the JSON output.

```powershell
.\.venv\Scripts\python.exe host/ai/predict_advanced_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

Create a waveform attribution plot, audit cross-dataset labels, analyze errors,
and run controlled research ablations:

```powershell
.\.venv\Scripts\python.exe host/ai/explain_advanced_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --class-name sinus_rhythm
.\.venv\Scripts\python.exe host/ai/audit_rhythm_label_alignment.py
.\.venv\Scripts\python.exe host/ai/analyze_advanced_rhythm_errors.py
.\.venv\Scripts\python.exe host/ai/run_advanced_rhythm_ablations.py
```

The 100, 250, and 500 Hz ablations are offline AI experiments only. They do not
select or finalize the pending hardware sampling rate. See
[`docs/AI_ADVANCED_BIPOLAR_RHYTHM_CLASSIFIER.md`](docs/AI_ADVANCED_BIPOLAR_RHYTHM_CLASSIFIER.md)
for the exact protocol, results, and limitations.

The selected full model reached 0.7900 macro average precision and 0.7898
exact-match accuracy on the Chapman held-out test, and 0.6108 macro average
precision and 0.6720 exact-match accuracy on PTB-XL fold 10. These are research
benchmark results, not clinical-performance claims.

## Machine-learning environment

The training environment uses Python 3.11 x64 and a project-local `.venv`.
Setup and verification instructions are documented in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Intended-use limitation

This repository contains an academic prototype, not a validated or certified
medical device. Future AI outputs will support professional analysis and will not
constitute autonomous diagnoses.
