import os
import tempfile
import unittest

from fastapi import BackgroundTasks, UploadFile

from app.api.routes import documents
from app.core.config import settings
from app.services import upload_jobs


class UploadBackgroundTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._original_upload_dir = settings.upload_dir
        self._temp_dir = tempfile.TemporaryDirectory()
        settings.upload_dir = self._temp_dir.name
        upload_jobs._jobs.clear()

    async def asyncTearDown(self):
        upload_jobs._jobs.clear()
        settings.upload_dir = self._original_upload_dir
        self._temp_dir.cleanup()

    async def test_upload_queues_background_processing(self):
        calls = []

        original_get_documents = documents.get_documents

        try:
            documents.get_documents = lambda user_id: []
            background_tasks = BackgroundTasks()

            def fake_add_task(func, *args, **kwargs):
                calls.append((func, args, kwargs))

            background_tasks.add_task = fake_add_task

            upload = UploadFile(filename="sample.pdf", file=tempfile.SpooledTemporaryFile())
            upload.file.write(b"%PDF-1.4 test content")
            upload.file.seek(0)

            response = await documents.upload_pdf(
                background_tasks=background_tasks,
                file=upload,
                user={"id": 123, "username": "tester"},
            )

            saved_path = os.path.join(settings.upload_dir, "tester", "sample.pdf")

            self.assertTrue(os.path.exists(saved_path))
            self.assertEqual(response.status, "queued")
            self.assertEqual(response.filename, "sample.pdf")
            self.assertTrue(response.job_id)
            self.assertEqual(response.chunks_created, 0)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], documents._run_document_processing)
            self.assertEqual(calls[0][1], (123, response.job_id, "sample.pdf", saved_path))
            self.assertEqual(upload_jobs.get_job(response.job_id)["status"], "queued")
        finally:
            documents.get_documents = original_get_documents


if __name__ == "__main__":
    unittest.main()
