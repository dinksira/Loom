"""
loom/normalize.py

Module for normalizing and cleaning text data.
"""

import re
import string
from typing import List
from nltk.corpus import stopwords

class Normalizer:
    """
    Normalizer class to process and clean text data.
    """

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """Normalize the text by cleaning and tokenizing."""
        # Lowercase the text
        text = raw_text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize the normalized text into words."""
        tokens = text.split()  # Simple whitespace-based tokenization
        return [token for token in tokens if token]  # Remove empty tokens

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]

    @classmethod
    def process(cls, raw_text: str) -> (str, List[str]):
        """Full normalization pipeline."""
        normalized_text = cls.normalize_text(raw_text)
        tokens = cls.tokenize(normalized_text)
        tokens = cls.remove_stopwords(tokens)
        return normalized_text, tokens