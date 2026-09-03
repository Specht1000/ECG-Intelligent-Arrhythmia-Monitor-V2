"""Mathematical relationships for bipolar limb leads.

The current acquisition boundary contains the two independent bipolar leads I
and II. Lead III is derived through Einthoven's law. Legacy 12-lead helpers are
retained for reproducibility of earlier experiments but are outside the current
four-electrode project scope. Values may use any unit if all inputs match.

This module does not perform electrode-potential conversion, calibration,
filtering, resampling, or signal-quality assessment.
"""

from collections import OrderedDict
from numbers import Real
from typing import Mapping, Tuple


BASE_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
BIPOLAR_LIMB_LEADS = ("I", "II", "III")
STANDARD_12_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _require_real(name: str, value: Real) -> None:
    if not isinstance(value, Real):
        raise TypeError("Lead {} must be a real number, got {!r}".format(name, value))


def derive_limb_leads(lead_i: Real, lead_ii: Real) -> Tuple[Real, Real, Real, Real]:
    """Return III, aVR, aVL and aVF from leads I and II.

    Formulas follow the standard Einthoven/Goldberger relationships:

    * III = II - I
    * aVR = -(I + II) / 2
    * aVL = I - II / 2
    * aVF = II - I / 2
    """

    _require_real("I", lead_i)
    _require_real("II", lead_ii)

    lead_iii = lead_ii - lead_i
    lead_avr = -(lead_i + lead_ii) / 2.0
    lead_avl = lead_i - lead_ii / 2.0
    lead_avf = lead_ii - lead_i / 2.0
    return lead_iii, lead_avr, lead_avl, lead_avf


def reconstruct_bipolar_limb_leads(
    independent_leads: Mapping[str, Real],
) -> "OrderedDict[str, Real]":
    """Return I, II, and derived III in canonical bipolar order."""

    missing = [name for name in ("I", "II") if name not in independent_leads]
    if missing:
        raise ValueError("Missing independent lead(s): {}".format(", ".join(missing)))
    lead_i = independent_leads["I"]
    lead_ii = independent_leads["II"]
    _require_real("I", lead_i)
    _require_real("II", lead_ii)
    return OrderedDict((("I", lead_i), ("II", lead_ii), ("III", lead_ii - lead_i)))


def reconstruct_12_leads(base_leads: Mapping[str, Real]) -> "OrderedDict[str, Real]":
    """Reconstruct one simultaneous 12-lead sample in canonical order.

    ``base_leads`` must provide I, II, and V1 through V6. Extra metadata keys are
    ignored so a decoded acquisition record can be passed directly.
    """

    missing = [name for name in BASE_LEADS if name not in base_leads]
    if missing:
        raise ValueError("Missing independent lead(s): {}".format(", ".join(missing)))

    for name in BASE_LEADS:
        _require_real(name, base_leads[name])

    lead_iii, lead_avr, lead_avl, lead_avf = derive_limb_leads(
        base_leads["I"], base_leads["II"]
    )

    return OrderedDict(
        (
            ("I", base_leads["I"]),
            ("II", base_leads["II"]),
            ("III", lead_iii),
            ("aVR", lead_avr),
            ("aVL", lead_avl),
            ("aVF", lead_avf),
            ("V1", base_leads["V1"]),
            ("V2", base_leads["V2"]),
            ("V3", base_leads["V3"]),
            ("V4", base_leads["V4"]),
            ("V5", base_leads["V5"]),
            ("V6", base_leads["V6"]),
        )
    )
