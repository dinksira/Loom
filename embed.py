"""
loom/embed.py

Module for generating TF-IDF embeddings for documents.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from .models import Document

class Embedder:
    """
    Embedder class to create TF-IDF embeddings for documents.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)

    def fit_transform(self, documents: List[Document]) -> Dict[int, List[float]]:
        """Fit the model and transform documents into TF-IDF vectors."""
        raw_texts = [doc.clean_text for doc in documents]
        tfidf_matrix = self.vectorizer.fit_transform(raw_texts)
        tfidf_array = tfidf_matrix.toarray()
        
        # Map document ID to its corresponding TF-IDF vector
        return {i: tfidf_array[i].tolist() for i in range(len(documents))}

    def get_feature_names(self) -> List[str]:
        """Get the feature names (terms) from the vectorizer."""
        return self.vectorizer.get_feature_names_out().tolist()