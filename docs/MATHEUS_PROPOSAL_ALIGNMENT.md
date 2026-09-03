# Alignment with Matheus Oliveira's proposal

## Reviewed source

This review covers all 16 pages of `anteprojeto_Matheus_Oliveira.pdf`. The PDF
proposes a four-electrode ECG prototype, analog conditioning, embedded digital
processing based on Pan-Tompkins, and research validation with public datasets.

## Adopted scope

The current project adopts the proposal's four-electrode body interface:

- RA: right-arm measurement electrode;
- LA: left-arm measurement electrode;
- LL: left-leg or lower-abdomen measurement electrode;
- RL: reference/return electrode.

Following the PFE author's later instruction, the software scope is narrower than
the proposal's complete frontal-plane set. The AI uses only the bipolar limb
leads I, II, and III. It excludes aVR, aVL, aVF, and V1 through V6.

```text
I   = LA - RA
II  = LL - RA
III = LL - LA = II - I
```

Only I and II are independent. Providing III to the model preserves the expected
three-lead representation but does not add independent information.

## AI interpretation of the proposal

Pan-Tompkins is appropriate as a future deterministic QRS and R-peak component.
It can support heart-rate, RR-interval, and rhythm-regularity features. It does
not by itself define the six-label classifier or establish that atrial
fibrillation and atrial flutter can be distinguished reliably.

The current AI therefore remains a computer-side, complete-exam research model.
It is trained from only I, II, and III in Chapman and externally evaluated from
the same channels in PTB-XL. MIT-BIH remains auxiliary because its channels are
modified leads and do not reproduce the final hardware input exactly.

## Differences that are not silently adopted

| Proposal item | Current project status |
| --- | --- |
| ARM Cortex-M4 platform | The confirmed board is NUCLEO-F072RB with a Cortex-M0 STM32F072RBT6. |
| AI and interface embedded in the device | The confirmed V2 AI and main interface run on the computer. |
| All six frontal-plane leads | The approved current scope is I, II, and III only. |
| INA114A and a second gain stage | Analog-front-end selection remains pending PCB review. |
| Fixed 60 Hz analog notch | Mains handling and filter parameters remain pending. |
| Approximately 150 Hz anti-alias cutoff | Sampling rate and anti-alias design must be selected together. |
| 0.05-100 Hz digital passband | Digital filtering remains pending acquisition and validation data. |
| Optocoupler-based protection | A complete isolation and patient-safety architecture is still required. |
| Display and keypad | Interface requirements remain pending; they are not assumed by the software. |

These differences are not criticisms of the proposal. They identify decisions
that changed later or that still require electrical design calculations,
component selection, and validation.

## Validation consequences

- Results from the earlier 12-lead model are historical and cannot represent the
  four-electrode prototype.
- The bipolar model must be compared on held-out Chapman and PTB-XL external data.
- Dataset ECGs are clinical recordings; they do not reproduce the final PCB's
  gain, noise, electrode placement, filtering, ADC, or timing behavior.
- Hardware-domain validation with reference signals remains mandatory.
- The prototype remains a research support tool and is not a certified medical
  device or autonomous diagnostic system.
