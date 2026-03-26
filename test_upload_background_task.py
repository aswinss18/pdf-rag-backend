import os
import tempfile
import unittest

from fastapi import UploadFile

from app.api.routes import documents
from app.core.config import settings
from app.services import upload_jobs


class UploadQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._original_upload_dir = settings.upload_dir
        self._temp_dir = tempfile.TemporaryDirectory()
        settings.upload_dir = self._temp_dir.name

    async def asyncTearDown(self):
        settings.upload_dir = self._original_upload_dir
        self._temp_dir.cleanup()

    async def test_upload_enqueues_rq_job(self):
        calls = []

        original_get_documents = documents.get_documents
        original_enqueue = documents.enqueue_document_processing

        try:
            documents.get_documents = lambda user_id: []

            def fake_enqueue(user_id, filename, file_path):
                calls.append((user_id, filename, file_path))
                return {
                    "job_id": "job-123",
                    "status": "queued",
                    "message": "Upload received. Chunking and embedding will start shortly.",
                }

            documents.enqueue_document_processing = fake_enqueue

            upload = UploadFile(filename="sample.pdf", file=tempfile.SpooledTemporaryFile())
            upload.file.write(b"%PDF-1.4 test content")
            upload.file.seek(0)

            response = await documents.upload_pdf(
                file=upload,
                user={"id": 123, "username": "tester"},
            )

            saved_path = os.path.join(settings.upload_dir, "tester", "sample.pdf")

            self.assertTrue(os.path.exists(saved_path))
            self.assertEqual(response.status, "queued")
            self.assertEqual(response.filename, "sample.pdf")
            self.assertEqual(response.job_id, "job-123")
            self.assertEqual(response.chunks_created, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], (123, "sample.pdf", saved_path))
        finally:
            documents.get_documents = original_get_documents
            documents.enqueue_document_processing = original_enqueue


if __name__ == "__main__":
    unittest.main()
