# Host Data Contract

## Independent inputs

The first internal host contract receives eight simultaneous leads:

```text
I, II, V1, V2, V3, V4, V5, V6
```

Every value in one sample must use the same unit and represent the same acquisition
instant. Microvolts are the recommended internal boundary unit, but this can only
be fixed after ADC and analog-front-end calibration is defined.

## Reconstructed output

The host produces this canonical order:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

The precordial leads are preserved, while III and the aV* limb leads are calculated
from I and II. This reconstruction assumes that the PCB supplies electrically
correct leads; it does not correct references, gain, offset, or channel skew.

## Verifiable invariants

```text
I + III = II
aVR + aVL + aVF = 0
```

These identities are useful software checks, but they do not replace hardware
calibration and signal-quality verification.
