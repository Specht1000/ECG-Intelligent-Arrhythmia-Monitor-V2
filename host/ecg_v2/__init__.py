"""Core host-side primitives for the ECG V2 research prototype."""

from .leads import BASE_LEADS, STANDARD_12_LEADS, derive_limb_leads, reconstruct_12_leads

__all__ = [
    "BASE_LEADS",
    "STANDARD_12_LEADS",
    "derive_limb_leads",
    "reconstruct_12_leads",
]
