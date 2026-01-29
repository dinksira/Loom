"""
loom/analyze.py

Analysis module for clustering documents, measuring similarity,
and detecting orphaned ideas.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .models import Cluster, AnalysisResult


class Analyzer:
    """
    Performs semantic analysis over document embeddings.
    """

    @staticmethod
    def cluster_documents(
        vectors: Dict[int, List[float]],
        num_clusters: int,
    ) -> List[Cluster]:
        """
        Cluster document embeddings using K-Means.

        Returns a list of Cluster objects containing document IDs,
        approximate keywords, and coherence scores.
        """
        if not vectors:
            return []

        doc_ids = list(vectors.keys())
        matrix = np.array(list(vectors.values()))

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(matrix)

        clusters: List[Cluster] = []

        for cluster_id in range(num_clusters):
            member_indices = np.where(labels == cluster_id)[0]
            member_docs = [doc_ids[i] for i in member_indices]

            coherence = Analyzer.compute_coherence_score(member_docs, vectors)
            keywords = Analyzer.derive_cluster_keywords(kmeans, cluster_id)

            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    document_ids=member_docs,
                    keywords=keywords,
                    coherence_score=coherence,
                )
            )

        return clusters

    @staticmethod
    def compute_coherence_score(
        cluster_docs: List[int],
        vectors: Dict[int, List[float]],
    ) -> float:
        """
        Compute mean pairwise cosine similarity for a cluster.

        Single-document clusters have zero coherence by definition.
        """
        if len(cluster_docs) < 2:
            return 0.0

        matrix = np.array([vectors[doc_id] for doc_id in cluster_docs])
        similarities = cosine_similarity(matrix)

        # Exclude self-similarity (diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        return float(np.mean(similarities[mask]))

    @staticmethod
    def derive_cluster_keywords(
        kmeans: KMeans,
        cluster_id: int,
        top_n: int = 10,
    ) -> List[str]:
        """
        Derive placeholder keywords from cluster centroids.

        NOTE:
        This does NOT represent true keywords unless the embedding
        space is explicitly tied to a vocabulary (e.g., TF-IDF).
        """
        centroid = kmeans.cluster_centers_[cluster_id]
        top_indices = np.argsort(centroid)[::-1][:top_n]

        return [f"dimension_{i}" for i in top_indices]

    @staticmethod
    def detect_orphans(
        vectors: Dict[int, List[float]],
        similarity_threshold: float,
    ) -> List[int]:
        """
        Detect orphan documents based on maximum similarity
        to any other document.
        """
        if len(vectors) < 2:
            return list(vectors.keys())

        doc_ids = list(vectors.keys())
        matrix = np.array(list(vectors.values()))

        similarities = cosine_similarity(matrix)

        orphan_ids: List[int] = []

        for i, doc_id in enumerate(doc_ids):
            # Ignore self-similarity
            other_similarities = np.delete(similarities[i], i)
            max_similarity = float(np.max(other_similarities))

            if max_similarity < similarity_threshold:
                orphan_ids.append(doc_id)

        return orphan_ids

    @staticmethod
    def analyze(
        vectors: Dict[int, List[float]],
        num_clusters: int,
        orphan_threshold: float,
    ) -> AnalysisResult:
        """
        Run full analysis pipeline: clustering + orphan detection.
        """
        clusters = Analyzer.cluster_documents(vectors, num_clusters)
        orphan_doc_ids = Analyzer.detect_orphans(vectors, orphan_threshold)

        return AnalysisResult(
            run_timestamp=datetime.utcnow(),
            document_count=len(vectors),
            clusters=clusters,
            orphan_doc_ids=orphan_doc_ids,
            orphan_threshold=orphan_threshold,
            num_clusters=num_clusters,
        )
