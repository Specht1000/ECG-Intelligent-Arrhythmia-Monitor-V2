# V2 Software Architecture

## Principle

Acquisition must remain deterministic and simple on the STM32. Heavy processing,
the user interface, storage, and AI run on the computer. This keeps the firmware
auditable and allows algorithms to change without modifying the acquisition chain.

## Blocks

### Acquisition PCB

The PCB is responsible for protection, analog conditioning, common-mode rejection,
A/D conversion, and disconnected-electrode indication according to components that
have not yet been selected. It must provide simultaneous channels or a sampling
strategy with known skew that is suitable for bipolar limb leads I and II. Lead
III may be measured or reconstructed as `II - I`; this remains a PCB decision.

### STM32 Nucleo

Responsibilities:

- configure and read the ADC or analog front end;
- group samples that belong to the same acquisition instant;
- assign a monotonic counter and timestamp;
- transport samples without diagnostic interpretation;
- transmit hardware state, lead-off information, and errors;
- count data loss and buffer overruns.

The controller board is the NUCLEO-F072RB with an STM32F072RBT6. The transport
remains a pending decision.

### Computer application

Planned pipeline:

```text
reception -> validation -> calibration -> signal-quality assessment
          -> filtering -> bipolar-lead validation/reconstruction
          -> visualization/recording -> AI windows
          -> assistive result with confidence and explanation
```

The current implementation includes the mathematical limb-lead relationship.
Filtering has not been implemented because the sampling rate, mains frequency,
and analog-front-end characteristics have not been confirmed.

### Artificial intelligence

The AI must consume calibrated data and maintain traceability between each
prediction, its signal segment, the model version, and the processing parameters.
The first engineering benchmark uses six rhythm labels, as recorded in
`PROJECT_CONTEXT.md`; the final specialist-approved clinical taxonomy remains
pending. The interface must allow an "inconclusive" result when signal quality or
model confidence is low.

## Minimum data in each sample block

- protocol version;
- block identifier or counter;
- first-sample index and monotonic timestamp;
- declared sampling rate;
- channel count and explicit channel order;
- raw or calibrated samples with a declared unit;
- disconnected-electrode or acquisition-quality mask;
- hardware status and loss counter;
- block integrity check.

The binary format and physical transport will only be defined after the analog
front end, ADC, acquisition topology, and computer interface have been selected.
