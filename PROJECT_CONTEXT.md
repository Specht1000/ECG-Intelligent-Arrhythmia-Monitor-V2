# Project Context — ECG V2

This file is the current source of truth for the project. The documents under
`docs/` mainly describe V1 and are historical references. If they conflict with
this file, this file takes precedence.

## Objective

Develop a second version of a low-cost bipolar limb-lead ECG system for the
fifth-year PFE.
The system must acquire cardiac signals, transmit them to a computer, and use
artificial intelligence to assist a cardiologist with arrhythmia analysis.

The system is an academic and research prototype. The AI must act as an assistant,
present interpretable results and uncertainty, and never replace the cardiologist's
diagnosis or decision. No clinical-use claim may be made without appropriate
validation and certification.

## Confirmed V2 decisions

- The microcontroller board is the **NUCLEO-F072RB**, populated with the
  **STM32F072RBT6** MCU.
- A **dedicated PCB**, developed in collaboration with another team member, will
  perform ECG analog acquisition and conditioning.
- The system will use **four electrodes**: RA, LA, LL, and RL. RL is the
  reference/return electrode and is not a diagnostic lead.
- The current system and AI scope contains only the **three bipolar limb leads**:
  I, II, and III. Augmented limb leads and precordial leads are excluded.
- The main processing, user interface, and AI will run on the computer.
- The AI will be a **cardiologist-support tool for arrhythmia analysis**, not an
  autonomous diagnostic system.
- The **Chapman-Shaoxing-Ningbo ECG database** is the primary dataset for the
  first arrhythmia-classification experiments. Only its I, II, and III channels
  are supplied to the current model.
- **PTB-XL 1.0.3** is the complementary dataset for patient-level validation and
  cross-dataset evaluation. Only its I, II, and III channels are supplied to the
  current model.
- The first V2 complete-exam rhythm classifier is a **six-label multi-label
  benchmark** covering sinus rhythm, sinus bradycardia, sinus tachycardia, sinus
  arrhythmia, atrial fibrillation, and atrial flutter. This initial engineering
  taxonomy does not replace the future specialist-approved clinical taxonomy.
- The **MIT-BIH Arrhythmia Database** is an auxiliary dataset for beat-level and
  R-peak experiments only. Its two modified-lead channels will not be used as
  evidence of final bipolar limb-lead model performance.
- V2 will use a new architecture. V1 components and limitations are not inherited
  automatically.

## Relationship between electrodes and leads

Four electrodes provide three bipolar limb leads. RA, LA, and LL are measurement
electrodes; RL is the reference/return electrode. The bipolar leads are:

```text
I   = LA - RA
II  = LL - RA
III = LL - LA = II - I
```

Only I and II are mathematically independent. Lead III may be acquired directly
or reconstructed from synchronized I and II samples, depending on the PCB. This
logical relationship does not finalize the PCB topology, isolation, driven-
reference circuit, or ADC channel implementation.

## Current functional architecture

```text
4 electrodes (RA, LA, LL, RL reference/return)
    -> patient protection + analog front end + ADC on the PCB
    -> STM32 Nucleo (synchronized acquisition, timestamps, and transport)
    -> computer (validation, filtering, bipolar-lead handling, UI, and storage)
    -> AI (analysis support, probabilities, uncertainty, and explainability)
    -> cardiologist
```

The firmware must not make diagnostic decisions. Its main responsibilities are
to acquire synchronized samples, preserve timing integrity, transmit electrode
and hardware status metadata, and report errors.

## AI and arrhythmias

The first six-label engineering benchmark has been approved. The final clinical
class list still requires specialist review. The five classes used in V1 (normal,
supraventricular, ventricular, fusion, and other) are references only and **do
not** define the V2 taxonomy.

The approved initial benchmark is a multi-label, complete-exam task with direct
Chapman SNOMED CT to PTB-XL SCP-ECG mappings:

| Benchmark label | Chapman SNOMED CT | PTB-XL SCP-ECG |
| --- | --- | --- |
| Sinus rhythm | 426783006 | SR |
| Sinus bradycardia | 426177001 | SBRAD |
| Sinus tachycardia | 427084000 | STACH |
| Sinus arrhythmia | 427393009 | SARRH |
| Atrial fibrillation | 164889003 | AFIB |
| Atrial flutter | 164890007 | AFLT |

The six-label engineering benchmark has now been trained with a reproducible
patient-separated protocol, per-class metrics, probability calibration,
experimental abstention, and integrated-gradients explanations. The following
still must be finalized before later clinical-model training and specialist
validation:

- the specialist-approved clinical task and time scale beyond the current
  complete-exam engineering benchmark;
- the final clinical taxonomy and labeling standard;
- datasets with licensing appropriate for the intended research and validation;
- clinically approved acceptance criteria, calibration, and abstention policy;
- the presentation and interpretation of explanations by the cardiologist;
- the specialist validation protocol.

## Pending decisions requiring confirmation

None of the following items may be finalized without consultation:

1. Pin/peripheral allocation and memory budget for the confirmed NUCLEO-F072RB.
2. Analog front end, ADC, number of simultaneous channels, isolation, and protection.
3. Signals delivered by the PCB and their electrical references.
4. Sampling rate, resolution, dynamic range, unit, and calibration.
5. STM32-to-PC interface, such as USB CDC, native USB, or Ethernet.
6. Binary framing, checksum/CRC, and protocol versioning mechanism.
7. Digital filters and parameters, including 50/60 Hz mains interference handling.
8. Signal-quality criteria and disconnected-electrode detection.
9. Final arrhythmia taxonomy and cross-dataset label harmonization.
10. Graphical interface, recording, and export requirements.
11. Power, isolation, and electrical-safety strategy.
12. Verification plan, validation plan, and prototype use limitations.

## Legacy references (V1)

- `docs/Report_ECG.pdf`: V1 report using an ESP32-S3, AD8232, ADS1115,
  three electrodes, approximately 250 Hz acquisition, and a CNN on the computer.
- `docs/DescECG1.pdf`: V1 progress summary.
- `docs/PresentationECG.pptx`: V1 presentation.

V1 results, including the reported overall accuracy of 90.63%, must not be used
as evidence of V2 performance because the architecture, lead count, and future
clinical task are different.

## Change rule

Decisions marked as confirmed may only be changed after consultation with the PFE
author. New decisions must be recorded in this file with their date and rationale.

## Decision log

- **2026-08-30 — Controller board:** NUCLEO-F072RB confirmed from the markings
  `NUCLEO-F072RB` and `NUF072RB$AU1` on the physical board. The matching MCU is
  STM32F072RBT6. Rationale: this is the board available for the V2 prototype.
- **2026-08-30 — AI datasets:** Chapman-Shaoxing-Ningbo confirmed as the primary
  dataset for initial 12-lead arrhythmia experiments. MIT-BIH retained only as an
  auxiliary beat-level dataset. Rationale: the primary model must use the 12
  standard leads produced by the V2 acquisition architecture.
- **2026-08-30 — Complementary validation dataset:** PTB-XL 1.0.3 added for
  patient-level validation and cross-dataset evaluation. Its 500 Hz records match
  the 10-second, 12-lead input structure used by the primary Chapman experiments.
- **2026-08-31 - Initial rhythm taxonomy:** a six-label, complete-exam,
  multi-label benchmark was approved for sinus rhythm, sinus bradycardia, sinus
  tachycardia, sinus arrhythmia, atrial fibrillation, and atrial flutter. Direct
  Chapman SNOMED CT and PTB-XL SCP-ECG mappings are recorded above. Rationale:
  these rhythm labels occur in both 12-lead datasets and support a first
  cross-dataset experiment without claiming to be the final clinical taxonomy.
- **2026-08-31 - Four-electrode bipolar scope:** the current acquisition and AI
  input were changed to four electrodes (RA, LA, LL, and RL reference/return) and
  the three bipolar limb leads I, II, and III. Augmented limb and precordial leads
  are excluded. Rationale: align the software with Matheus Oliveira's proposal
  while following the PFE author's explicit instruction to use bipolar leads only.
- **2026-09-03 - Advanced bipolar research benchmark:** Chapman training data and
  PTB-XL folds 1-8 were combined for the six-label model; validation combines the
  established Chapman validation split with PTB-XL fold 9, while Chapman held-out
  data and PTB-XL fold 10 remain separate tests. The selected model reached 0.7900
  and 0.6108 macro average precision on those tests, respectively. Controlled
  short-training ablations ranked I/II at 100 Hz above explicit I/II/III at 100,
  250, and 500 Hz. Rationale: measure cross-dataset generalization and input-cost
  trade-offs without changing the confirmed four-electrode bipolar scope. The
  sample rates, quality gate, thresholds, and abstention rules remain offline
  research settings and do not finalize hardware or clinical requirements.
