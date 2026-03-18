from typing import Optional 

# BCS-DBT LABEL SCHEMA (VERSION 1) 


BCS_DBT_POSITIVE = {
    "cancer",
    "malignant",
    "biopsy_proven_cancer",
}

BCS_DBT_NEGATIVE = {
    "normal",
    "benign",
    "benign_biopsy",
    "no_finding",
}

BCS_DBT_IGNORE = {
    "actionable",
    "suspicious",
    "probably_benign",
    "callback",
    "unclear",
}

def normalize_raw_label(raw_label: str) -> str: 
    """
    Normalize a raw label string for robust matching: 
    """
    return (raw_label or "").strip().lower()

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