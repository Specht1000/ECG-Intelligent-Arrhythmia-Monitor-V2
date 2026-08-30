# Project Context — ECG V2

This file is the current source of truth for the project. The documents under
`docs/` mainly describe V1 and are historical references. If they conflict with
this file, this file takes precedence.

## Objective

Develop a second version of a low-cost 12-lead ECG system for the fifth-year PFE.
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
- The system will use **10 electrodes**: RA, LA, RL, LL, and V1 through V6.
- The system will work with the **12 standard leads**: I, II, III, aVR, aVL, aVF,
  and V1 through V6.
- The main processing, user interface, and AI will run on the computer.
- The AI will be a **cardiologist-support tool for arrhythmia analysis**, not an
  autonomous diagnostic system.
- The **Chapman-Shaoxing-Ningbo 12-lead ECG database** is the primary dataset for
  the first arrhythmia-classification experiments.
- **PTB-XL 1.0.3** is the complementary 12-lead dataset for patient-level
  validation, cross-dataset evaluation, and later combined training after label
  harmonization.
- The **MIT-BIH Arrhythmia Database** is an auxiliary dataset for beat-level and
  R-peak experiments only. Its two-channel recordings will not be used as evidence
  of final 12-lead model performance.
- V2 will use a new architecture. V1 components and limitations are not inherited
  automatically.

## Relationship between electrodes and leads

Ten electrodes do not mean ten leads. The RL electrode is normally used as a
reference/return electrode, and the 12 leads are constructed from limb and
precordial measurements.

When the PCB provides I, II, and V1 through V6, the remaining limb leads will be
calculated on the computer:

```text
III = II - I
aVR = -(I + II) / 2
aVL = I - II / 2
aVF = II - I / 2
```

This logical interface does not define the PCB's electrical topology. The PCB
designer must confirm whether the precordial outputs are already referenced to
the Wilson central terminal and exactly which signals reach the ADC.

## Current functional architecture

```text
10 electrodes
    -> patient protection + analog front end + ADC on the PCB
    -> STM32 Nucleo (synchronized acquisition, timestamps, and transport)
    -> computer (validation, filtering, 12-lead reconstruction, UI, and storage)
    -> AI (analysis support, probabilities, uncertainty, and explainability)
    -> cardiologist
```

The firmware must not make diagnostic decisions. Its main responsibilities are
to acquire synchronized samples, preserve timing integrity, transmit electrode
and hardware status metadata, and report errors.

## AI and arrhythmias

The final class list has not been approved. The five classes used in V1 (normal,
supraventricular, ventricular, fusion, and other) are references only and **do
not** define the V2 taxonomy.

The following must be defined before training:

- the clinical task: beat, rhythm, event, or complete-exam classification;
- the taxonomy and labeling standard;
- licensed 12-lead ECG datasets;
- patient-level separation between training, validation, and test sets;
- per-class metrics, probability calibration, and abstention policy;
- the explainability method and its presentation to the cardiologist;
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
