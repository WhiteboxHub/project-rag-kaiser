"""Test script to query the RAG pipeline locally."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rag.query_pipeline import RAGPipeline


def main():
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(top_k=3)

    # Test queries
    test_queries = [
        "What are the principles of responsibility?",
        "How do I file a claim?",
        "What is covered by Kaiser insurance?"
    ]

    for question in test_queries:
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")

        result = pipeline.query(question)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nRETRIEVED {result['num_chunks']} CHUNKS:")
        for i, (chunk, score) in enumerate(zip(result['context'], result.get('scores', [])), 1):
            print(f"\n[Chunk {i}] (Relevance: {score:.3f})")
            print(f"{chunk[:200]}...")


if __name__ == "__main__":
    main()
