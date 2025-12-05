# # project-rag-kaiser/scripts/run_ingestion.py
import sys
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.ingestion import IngestionDocument

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def ingest_doc(pipeline, doc: IngestionDocument):
    logger.info("Ingesting %s ...", doc.file_path)
    try:
        result = pipeline.ingest(doc)
        logger.info("Ingested %s -> %s", doc.file_path, result)
        return result
    except Exception as e:
        logger.exception("Failed to ingest %s", doc.file_path)
        return {"file": doc.file_path, "status": "error", "error": str(e)}

def main(parallel: int = 1):
    pipeline = IngestionPipeline()

    doc_paths = [
        "data/kaiser/principlesofresponsibility-en.pdf",
        "data/kaiser/evidence-of-coverage-special-needs-eae-ncal.pdf",
        "data/kaiser/member-guide-wa-en.pdf",
    ]

    docs = []
    for p in doc_paths:
        path = Path(p)
        if not path.exists():
            logger.error("File does not exist: %s", p)
            continue
        docs.append(IngestionDocument(source="kaiser_principles", file_path=str(path)))

    if not docs:
        logger.error("No valid documents to ingest. Exiting.")
        return 1

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(ingest_doc, pipeline, d): d for d in docs}
            for fut in as_completed(futures):
                _ = fut.result()
    else:
        for d in docs:
            ingest_doc(pipeline, d)

    return 0

if __name__ == "__main__":
    raise SystemExit(main(parallel=1))
