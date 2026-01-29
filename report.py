"""
loom/report.py

Module for generating human-readable reports from analysis results.
"""

from typing import List
from .models import AnalysisResult, TraceResult

class Reporter:
    """
    Reporter class for generating analysis reports.
    """

    @staticmethod
    def generate_analysis_report(result: AnalysisResult, output_path: str):
        """Generate a report summarizing the analysis results."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Report - {result.run_timestamp}\n")
            f.write(f"Total Documents Analyzed: {result.document_count}\n")
            f.write(f"Detected Clusters: {len(result.clusters)}\n")
            f.write(f"Orphan Documents: {len(result.orphan_doc_ids)}\n\n")
            
            f.write("Clusters:\n")
            for cluster in result.clusters:
                f.write(f"Cluster ID: {cluster.cluster_id}\n")
                f.write(f"Keywords: {', '.join(cluster.keywords)}\n")
                f.write(f"Document IDs: {', '.join(map(str, cluster.document_ids))}\n")
                f.write(f"Coherence Score: {cluster.coherence_score}\n\n")
            
            f.write("Orphan Document IDs:\n")
            f.write(", ".join(map(str, result.orphan_doc_ids)) + "\n")
            
            # Placeholder for future reporting enhancements
            f.write("\n[Further enhancements can be added here]\n")

    @staticmethod
    def trace_insight(insight: TraceResult) -> str:
        """Generate text for tracing an insight to its source."""
        evidence_text = "\n".join(f"{ev['doc_path']}: {ev['relevant_excerpt']} (Score: {ev['score']})"
                                   for ev in insight.evidence)
        return f"{insight.insight_type} - {insight.insight_description}\nEvidence:\n{evidence_text}\n"