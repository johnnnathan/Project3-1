# collaboration_mapping.py

"""
Rule-based collaboration mapping for NTU RGB+D 60 action classes.

This file provides:
1. Discrete collaboration levels: low / medium / high
2. A continuous collaboration score in [0, 1]

This is a transparent baseline used before EDMO collaboration
annotations are available.
"""

from typing import List


# =============================
# NTU-aware keyword groupings
# =============================

HIGH_COLLAB_KEYWORDS: List[str] = [
    "shaking hands",
    "hugging",
    "giving object",
    "receiving object",
    "handshake",
    "pat on the back",
    "walk towards",
]

MEDIUM_COLLAB_KEYWORDS: List[str] = [
    "standing together",
    "talking",
    "conversation",
    "approaching",
]

LOW_COLLAB_KEYWORDS: List[str] = [
    "sitting down",
    "standing up",
    "reading",
    "writing",
    "typing",
    "using phone",
    "drinking",
    "eating",
    "walking",
]

NEGATIVE_COLLAB_KEYWORDS: List[str] = [
    "punching",
    "slapping",
    "kicking",
    "pushing",
    "hit",
    "walking apart",
]


# =============================
# Discrete collaboration level
# =============================

def map_action_to_collaboration_level(action_label: str) -> str:
    """
    Map an NTU action label to a collaboration level:
    'high', 'medium', or 'low'.
    """
    label = action_label.lower()

    if any(k in label for k in HIGH_COLLAB_KEYWORDS):
        return "high"

    if any(k in label for k in MEDIUM_COLLAB_KEYWORDS):
        return "medium"

    return "low"


# =============================
# Continuous collaboration score
# =============================

def action_to_collab_score(action_label: str) -> float:
    """
    Return a collaboration score in [0, 1].

    Interpretation:
    - 0.9 : strong collaborative intent
    - 0.6 : weak or ambiguous collaboration
    - 0.1 : individual / task-focused
    - 0.0 : conflict or anti-collaboration
    """
    label = action_label.lower()

    if any(k in label for k in NEGATIVE_COLLAB_KEYWORDS):
        return 0.0

    if any(k in label for k in HIGH_COLLAB_KEYWORDS):
        return 0.9

    if any(k in label for k in MEDIUM_COLLAB_KEYWORDS):
        return 0.6

    if any(k in label for k in LOW_COLLAB_KEYWORDS):
        return 0.1

    # Fallback for uncommon NTU actions
    return 0.3
