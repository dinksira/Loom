"""
loom/analyze.py

Module for analyzing document clusters, similarities, and orphan detection.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from .models import Cluster, AnalysisResult

class Analyzer:
    """
    Analyzer class to conduct analysis on document embeddings.
    """

    @staticmethod
    def cluster_documents(vectors: Dict[int, List[float]], num_clusters: int) -> List[Cluster]:
        """Cluster documents using K-means and return cluster information."""
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        doc_ids = list(vectors.keys())
        vector_array = np.array(list(vectors.values()))
        kmeans.fit(vector_array)
        
        clusters = []
        for cluster_id in range(num_clusters):
            cluster_docs = [doc_ids[i] for i in range(len(doc_ids)) if kmeans.labels_[i] == cluster_id]
            keywords = Analyzer.get_cluster_keywords(kmeans, cluster_id, vector_array)
            coherence_score = Analyzer.compute_coherence_score(cluster_docs, vectors)
            clusters.append(Cluster(cluster_id=cluster_id, keywords=keywords, document_ids=cluster_docs, 
                                   coherence_score=coherence_score))

        return clusters

    @staticmethod
    def compute_coherence_score(cluster_docs: List[int], vectors: Dict[int, List[float]]) -> float:
        """Compute average similarity score for documents in a cluster."""
        if len(cluster_docs) < 2:
            return 0.0
        similarities = cosine_similarity([vectors[doc] for doc in cluster_docs])
        return np.mean(similarities[~np.eye(similarities.shape[0], dtype=bool)])  # Exclude diagonal

    @staticmethod
    def get_cluster_keywords(kmeans, cluster_id: int, vector_array: np.ndarray) -> List[str]:
        """Get top keywords for a given cluster based on centroids."""
        centroid = kmeans.cluster_centers_[cluster_id]
        order_centroids = centroid.argsort()[::-1]
        return [f"feature_{index}" for index in order_centroids[:10]]  # Top 10 keywords

    @staticmethod
    def detect_orphans(vectors: Dict[int, List[float]], threshold: float) -> List[int]:
        """Detect documents with low connectivity (orphan documents)."""
        similarities = cosine_similarity(np.array(list(vectors.values())))
        max_similarities = np.max(similarities, axis=1)
        return [list(vectors.keys())[i] for i in range(len(max_similarities)) if max_similarities[i] < threshold]

    @staticmethod
    def analyze(vectors: Dict[int, List[float]], num_clusters: int, orphan_threshold: float) -> AnalysisResult:
        """Conduct full analysis, clustering and orphan detection."""
        clusters = Analyzer.cluster_documents(vectors, num_clusters)
        orphan_doc_ids = Analyzer.detect_orphans(vectors, orphan_threshold)
        return AnalysisResult(run_timestamp=datetime.now(), document_count=len(vectors),
                              clusters=clusters, orphan_doc_ids=orphan_doc_ids,
                              orphan_threshold=orphan_threshold, num_clusters=num_clusters)