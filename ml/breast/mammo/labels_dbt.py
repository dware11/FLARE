from typing import Optional
import re

# BCS-DBT LABEL SCHEMA (VERSION 1)
# ------------------------------------------------------------------
# TCIA BCS-DBT case labels are commonly:
#   - Normal
#   - Actionable
#   - Biopsy-proven benign
#   - Biopsy-proven cancer
#
# This file maps raw labels to:
#   - 1 positive  (biopsy-proven cancer)
#   - 0 negative  (normal + biopsy-proven benign)
#   - -1 ignore   (actionable; ambiguous without confirmed cancer)
# ------------------------------------------------------------------


def _canonicalize_bcs_dbt_label(raw_label: str) -> str:
    """
    Canonicalize BCS-DBT label strings to be robust to:
      - casing
      - whitespace
      - hyphens vs underscores
      - extra punctuation
    Example:
      "Biopsy-proven cancer" -> "biopsy_proven_cancer"
    """
    s = (raw_label or "").strip().lower()
    # Treat hyphens/underscores as spaces, then normalize whitespace.
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    return s


# After canonicalization:
BCS_DBT_POSITIVE = {
    "biopsy_proven_cancer",
}

BCS_DBT_NEGATIVE = {
    "normal",
    "biopsy_proven_benign",
}

BCS_DBT_IGNORE = {
    "actionable",
}


def normalize_raw_label(raw_label: str) -> str:
    """Backward-compatible wrapper for older code paths."""
    return _canonicalize_bcs_dbt_label(raw_label)

def map_bcs_dbt_label(raw_label: str) -> int: 
    """
    Map a BCS-DBT raw label string to: 
    - 1 = positive (confirmed cancer) 
    - 0 = negative (clearly non-cancer) 
    - -1 = ignore 
    """

    norm = normalize_raw_label(raw_label)

    if norm in BCS_DBT_POSITIVE:
        return 1
    if norm in BCS_DBT_NEGATIVE:
        return 0
    if norm in BCS_DBT_IGNORE:
        return -1

    return -1

def is_valid_label(y: int) -> bool: 
    """ Return True if y is a usable training label (0 or 1) """ 
    return y in (0,1) 


def describe_bcs_dbt_schema() -> str: 
    """
    Human-readable description for logging / debugging.
    Useful to print once at preprocessing start.
    """
    return (
        "BCS-DBT v1 label schema:\n"
        f" POSITIVE (1): {sorted(BCS_DBT_POSITIVE)}\n"
        f" NEGATIVE (0): {sorted(BCS_DBT_NEGATIVE)}\n"
        f" IGNORE (-1): {sorted(BCS_DBT_IGNORE)}\n"
        " Unknown labels -> IGNORE (-1) by default.\n"
    )