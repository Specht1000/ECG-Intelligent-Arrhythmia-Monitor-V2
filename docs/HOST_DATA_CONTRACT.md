# Host Data Contract

## Independent inputs

The current internal host contract receives two simultaneous bipolar limb leads:

```text
I, II
```

Every value in one sample must use the same unit and represent the same acquisition
instant. Microvolts are the recommended internal boundary unit, but this can only
be fixed after ADC and analog-front-end calibration is defined.

## Bipolar output

The host produces this canonical order for the AI:

```text
I, II, III
```

Lead III is calculated as `II - I` when it is not delivered directly. The
augmented limb leads and all precordial leads are outside the current scope. This
contract assumes that the PCB supplies electrically correct I and II signals; it
does not correct electrode references, gain, offset, or channel skew.

## Verifiable invariants

```text
I + III = II
```

These identities are useful software checks, but they do not replace hardware
calibration and signal-quality verification.
