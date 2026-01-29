"""
loom/cli.py

Command-line interface for Loom.
"""

import argparse
from pathlib import Path
from .ingest import Ingestor
from .normalize import Normalizer
from .embed import Embedder
from .analyze import Analyzer
from .report import Reporter
from .storage import Storage
from .config import DB_PATH, NUM_CLUSTERS, ORPHAN_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="Loom: Personal Document Analysis Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_parser.add_argument("directory", type=Path, help="Directory containing documents")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze ingested documents")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate analysis reports")
    report_parser.add_argument("output", type=Path, help="Output file for the report")

    args = parser.parse_args()

    # Storage setup
    storage = Storage(DB_PATH)

    if args.command == "ingest":
        documents = Ingestor.ingest_from_directory(args.directory)
        for doc in documents:
            storage.insert_document(doc)

    elif args.command == "analyze":
        vectors = {doc.id: doc.vector for doc in storage.get_all_documents()}  # Assuming a vector field is added
        analysis_result = Analyzer.analyze(vectors, NUM_CLUSTERS, ORPHAN_THRESHOLD)
        storage.store_analysis_result(analysis_result)

    elif args.command == "report":
        result = storage.get_analysis_result()  # Assume a method for fetching the latest analysis
        Reporter.generate_analysis_report(result, str(args.output))

    storage.close()

if __name__ == "__main__":
    main()