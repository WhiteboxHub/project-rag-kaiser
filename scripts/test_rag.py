# project-rag-kaiser/scripts/test_rag.py
import argparse
import logging
import warnings
import sys
from pathlib import Path
from rag.query_pipeline import RAGPipeline
from urllib3.exceptions import NotOpenSSLWarning

# optionally suppress the LibreSSL warning (development only)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

 

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--question", "-q", type=str, help="Single question to ask")
    grp.add_argument("--file", "-f", type=Path, help="Path to file with one question per line")
    p.add_argument("--top_k", "-k", type=int, default=3, help="Number of chunks to retrieve")
    p.add_argument("--verbose", "-v", action="store_true", help="Show retrieved chunks and scores")
    p.add_argument("--default", action="store_true", help="Run built-in default test queries")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return p.parse_args()


def load_questions_from_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Initializing RAG Pipeline (top_k=%d)...", args.top_k)
    pipeline = RAGPipeline(top_k=args.top_k)

    if args.default:
        questions = [
            "What are the principles of responsibility?",
            "How do I file a claim?",
            "What is covered by Kaiser insurance?"
        ]
    elif args.file:
        questions = load_questions_from_file(args.file)
    elif args.question:
        questions = [args.question]
    else:
        logger.error("No question provided. Use --question, --file, or --default")
        return 2

    for question in questions:
        logger.info("QUESTION: %s", question)
        try:
            result = pipeline.query(question)
        except Exception as e:
            logger.exception("Query failed for question: %s", question)
            continue

        answer = result.get("answer", "").strip()
        if args.verbose:
            print("\n" + "=" * 80)
            print(f"QUESTION: {question}")
            print("=" * 80)
            print("\nANSWER:\n")
            print(answer)
            print(f"\nRETRIEVED {result.get('num_chunks', 0)} CHUNKS:")
            for i, (chunk, score) in enumerate(zip(result.get('context', []), result.get('scores', [])), 1):
                print(f"\n[Chunk {i}] (Relevance: {score:.3f})")
                print(chunk[:400].replace("\n", " ").strip() + "...")
        else:
            # quiet: only print the answer
            print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
