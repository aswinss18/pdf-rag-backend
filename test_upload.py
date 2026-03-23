import sys
import logging
from app.services.internals.rag_pipeline import process_pdf
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

file_path = "uploaded/email.pdf"

try:
    process_pdf(file_path)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
