"""
loom/config.py

Configuration settings for the Loom application.
These constants control various aspects of the application behavior.
"""

import os
from pathlib import Path

# Data folder for document storage
DATA_DIR = Path(os.getenv("LOOM_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database file path
DB_PATH = DATA_DIR / "loom.db"

# Report output directory
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Embedding settings
TF_IDF_MIN_DF = 2  # Minimum document frequency for terms
TF_IDF_MAX_DF = 0.95  # Maximum document frequency for terms

# Clustering settings
NUM_CLUSTERS = 10  # Default number of clusters for K-means
ORPHAN_THRESHOLD = 0.1  # Threshold for low-document connectivity

# NLP settings
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"  # Custom stopwords file, if needed