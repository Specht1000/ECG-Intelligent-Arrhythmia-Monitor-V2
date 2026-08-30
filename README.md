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

## Repository structure

```text
Docs/                   documentation and references
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

## Machine-learning environment

The training environment uses Python 3.11 x64 and a project-local `.venv`.
Setup and verification instructions are documented in
[`Docs/ENVIRONMENT.md`](Docs/ENVIRONMENT.md).

## Intended-use limitation

This repository contains an academic prototype, not a validated or certified
medical device. Future AI outputs will support professional analysis and will not
constitute autonomous diagnoses.
