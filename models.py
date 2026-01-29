"""
loom/models.py

Data classes representing core domain objects.
These are plain data containers with no behavior beyond validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib


@dataclass
class Document:
    """
    Represents a single ingested document.
    
    Invariants:
    - path is always absolute
    - content_hash is SHA-256 of raw content (for change detection)
    - created_at reflects file mtime, not ingest time
    """
    path: Path
    raw_text: str
    content_hash: str
    created_at: datetime
    file_type: str  # 'markdown', 'txt', 'pdf'
    
    # Populated after normalization
    clean_text: str = ""
    tokens: list[str] = field(default_factory=list)
    word_count: int = 0
    
    # Assigned after embedding/analysis
    doc_id: Optional[int] = None
    cluster_id: Optional[int] = None
    
    @classmethod
    def compute_hash(cls, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def __post_init__(self):
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        self.path = self.path.resolve()


@dataclass
class Cluster:
    """
    Represents a topic cluster.
    
    Keywords are the top TF-IDF terms that characterize this cluster.
    This provides explainability: users can see WHY documents grouped together.
    """
    cluster_id: int
    keywords: list[str]  # Top N terms by centroid weight
    document_ids: list[int]
    coherence_score: float  # Average intra-cluster similarity
    
    def __len__(self):
        return len(self.document_ids)


@dataclass
class SimilarityEdge:
    """
    Represents similarity between two documents.
    Stored explicitly for traceability.
    """
    doc_id_a: int
    doc_id_b: int
    similarity: float
    shared_terms: list[str]  # Top contributing terms


@dataclass 
class AnalysisResult:
    """
    Complete analysis output for a corpus.
    This is the primary artifact produced by Loom.
    """
    run_timestamp: datetime
    document_count: int
    clusters: list[Cluster]
    orphan_doc_ids: list[int]  # Documents with low connectivity
    time_buckets: dict[str, list[int]]  # period -> doc_ids
    
    # Thresholds used (for reproducibility)
    orphan_threshold: float
    num_clusters: int


@dataclass
class TraceResult:
    """
    Traceback from an insight to source evidence.
    Answers: "Why did Loom say X?"
    """
    insight_type: str  # 'cluster', 'orphan', 'similarity'
    insight_description: str
    evidence: list[dict]  # List of {doc_path, relevant_excerpt, score}