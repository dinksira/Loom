"""
loom/config.py

Central configuration for the Loom application.

This module defines defaults and environment overrides for
storage, analysis, and NLP behavior. Side effects are limited
and intentional.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

BASE_DATA_DIR = Path(os.getenv("LOOM_DATA_DIR", "data")).resolve()

DB_PATH = BASE_DATA_DIR / "loom.db"
REPORTS_DIR = BASE_DATA_DIR / "reports"
STOPWORDS_FILE = BASE_DATA_DIR / "stopwords.txt"


def ensure_directories() -> None:
    """
    Ensure required directories exist.

    Called explicitly at application startup to avoid
    hidden import-time side effects.
    """
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Text processing / NLP
# ---------------------------------------------------------------------------

# Ignore terms that appear in fewer than this many documents
TF_IDF_MIN_DOC_FREQ = 2

# Ignore terms that appear in more than this fraction of documents
TF_IDF_MAX_DOC_FREQ = 0.95


# ---------------------------------------------------------------------------
# Clustering / analysis
# ---------------------------------------------------------------------------

# Default number of clusters for K-Means
# NOTE: This is a heuristic, not a truth.
DEFAULT_NUM_CLUSTERS = 10

# Maximum cosine similarity threshold below which a document
# is considered an "orphan"
ORPHAN_SIMILARITY_THRESHOLD = 0.10
