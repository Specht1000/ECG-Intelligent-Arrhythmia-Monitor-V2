# ECG V2 — PFE

Software foundation for a 12-lead ECG using an STM32 Nucleo, a dedicated
acquisition PCB, and AI-assisted analysis on a computer.

Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) first. It contains the current
decisions and the pending items that must not be resolved by assumption.

## Current status

- V2 context separated from the legacy V1 documentation;
- functional architecture and responsibility boundaries documented;
- 12-lead reconstruction from I, II, and V1 through V6 implemented on the host;
- unit tests for Einthoven's identity and the augmented limb leads;
- initial logical STM32-to-PC contract documented without fixing the physical
  transport.
- reproducible structural and label audits for Chapman-Shaoxing-Ningbo and
  PTB-XL, with machine-readable reports under `artifacts/dataset_audits/`.
- an experimental PTB-XL normal-versus-abnormal baseline with official,
  patient-separated folds and reproducible evaluation artifacts.
- an experimental six-label, multi-label rhythm classifier trained on Chapman
  and evaluated on held-out Chapman data and PTB-XL official fold 10.

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

Plot and save a random 12-lead ECG from Chapman or PTB-XL:

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

## Train the six-label rhythm classifier

The first approved V2 rhythm benchmark covers sinus rhythm, sinus bradycardia,
sinus tachycardia, sinus arrhythmia, atrial fibrillation, and atrial flutter. It
trains on Chapman and uses PTB-XL fold 10 as a strictly external evaluation set.

```powershell
.\.venv\Scripts\python.exe host/ai/train_rhythm_classifier.py
```

Run inference on one compatible 12-lead PTB-XL `records100` record:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
```

See [`docs/AI_RHYTHM_CLASSIFIER.md`](docs/AI_RHYTHM_CLASSIFIER.md) for label
mappings, the evaluation protocol, per-class results, and limitations.

Analyze per-class errors, calibration, demographic subgroups, threshold
proximity, and Chapman-to-PTB-XL domain shift:

```powershell
.\.venv\Scripts\python.exe host/ai/analyze_rhythm_errors.py
```

Create an experimental integrated-gradients explanation for one class:

```powershell
.\.venv\Scripts\python.exe host/ai/explain_rhythm_prediction.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --class-name sinus_rhythm
```

See
[`docs/AI_RHYTHM_ERROR_ANALYSIS.md`](docs/AI_RHYTHM_ERROR_ANALYSIS.md) for the
findings and interpretation limits.

## Machine-learning environment

The training environment uses Python 3.11 x64 and a project-local `.venv`.
Setup and verification instructions are documented in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Intended-use limitation

This repository contains an academic prototype, not a validated or certified
medical device. Future AI outputs will support professional analysis and will not
constitute autonomous diagnoses.
