import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ingestion.pipeline_new import IngestionPipeline
from app.schemas.ingestion import IngestionDocument


def main():
    pipeline = IngestionPipeline()

    docs = [
        IngestionDocument(
            source="kaiser_principles",
            file_path="data/kaiser/principlesofresponsibility-en.pdf"
        ),
        IngestionDocument(
            source="kaiser_principles",
            file_path="data/kaiser/evidence-of-coverage-special-needs-eae-ncal.pdf"
        ),
        IngestionDocument(
            source="kaiser_principles",
            file_path="data/kaiser/member-guide-wa-en.pdf"
        )
    ]

    # Ingest each document one by one
    for doc in docs:
        result = pipeline.ingest(doc)
        print(result)


if __name__ == "__main__":
    main()
