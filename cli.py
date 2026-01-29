"""
loom/cli.py

Command-line interface for Loom.
"""

import argparse
import sys
from pathlib import Path

from .ingest import Ingestor
from .normalize import Normalizer
from .embed import Embedder
from .analyze import Analyzer
from .report import Reporter
from .storage import Storage
from .config import DB_PATH, NUM_CLUSTERS, ORPHAN_THRESHOLD


def ingest_command(directory: Path, storage: Storage) -> None:
    documents = Ingestor.ingest_from_directory(directory)

    if not documents:
        print("No documents found to ingest.")
        return

    for doc in documents:
        storage.insert_document(doc)

    print(f"Ingested {len(documents)} documents.")


def analyze_command(storage: Storage) -> None:
    documents = storage.get_all_documents()

    if not documents:
        print("No documents available for analysis. Run `ingest` first.")
        return

    normalizer = Normalizer()
    embedder = Embedder()

    normalized_docs = [normalizer.normalize(doc) for doc in documents]
    vectors = embedder.embed_documents(normalized_docs)

    analysis_result = Analyzer.analyze(
        vectors=vectors,
        num_clusters=NUM_CLUSTERS,
        orphan_threshold=ORPHAN_THRESHOLD,
    )

    storage.store_analysis_result(analysis_result)

    print("Analysis completed successfully.")


def report_command(output_path: Path, storage: Storage) -> None:
    result = storage.get_latest_analysis()

    if result is None:
        print("No analysis results found. Run `analyze` first.")
        return

    Reporter.generate_analysis_report(result, output_path)
    print(f"Report written to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loom — a local-first tool for observing patterns in personal documents"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents from a directory",
    )
    ingest_parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing documents to ingest",
    )

    subparsers.add_parser(
        "analyze",
        help="Analyze ingested documents",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate an analysis report",
    )
    report_parser.add_argument(
        "output",
        type=Path,
        help="Output file path for the report",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    storage = Storage(DB_PATH)

    try:
        if args.command == "ingest":
            ingest_command(args.directory, storage)

        elif args.command == "analyze":
            analyze_command(storage)

        elif args.command == "report":
            report_command(args.output, storage)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        storage.close()


if __name__ == "__main__":
    main()
