import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.core.auth import get_current_user, hash_password
from app.core.config import settings
from app.db.sqlite_store import create_user, init_database
from app.db.vector_store import add_embeddings, clear_documents
from app.main import create_app


class DocumentsRouteTests(unittest.TestCase):
    def setUp(self):
        self._original_sqlite_db_path = settings.sqlite_db_path
        self._temp_dir = tempfile.TemporaryDirectory()
        settings.sqlite_db_path = os.path.join(self._temp_dir.name, "test.db")
        init_database()

        self.user = create_user("tester", hash_password("password123"))
        self.app = create_app()
        self.app.dependency_overrides[get_current_user] = lambda: {"id": self.user["id"], "username": self.user["username"]}
        self.client = TestClient(self.app)

        chunks = [
            {"doc": "alpha.pdf", "page": 1, "text": "a", "chunk_index": 0},
            {"doc": "alpha.pdf", "page": 2, "text": "b", "chunk_index": 1},
            {"doc": "alpha.pdf", "page": 2, "text": "c", "chunk_index": 2},
            {"doc": "beta.pdf", "page": 5, "text": "d", "chunk_index": 0},
        ]
        embeddings = [[0.0] * 1536 for _ in chunks]
        add_embeddings(self.user["id"], chunks[:3], embeddings[:3])
        add_embeddings(self.user["id"], chunks[3:], embeddings[3:])

    def tearDown(self):
        clear_documents(self.user["id"])
        self.app.dependency_overrides.clear()
        settings.sqlite_db_path = self._original_sqlite_db_path
        self._temp_dir.cleanup()

    def test_list_documents_returns_summary_without_mutation_error(self):
        response = self.client.get("/documents")
        result = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["total_chunks"], 4)
        self.assertEqual(result["documents"]["alpha.pdf"]["chunk_count"], 3)
        self.assertEqual(result["documents"]["alpha.pdf"]["pages"], [1, 2])
        self.assertEqual(result["documents"]["alpha.pdf"]["page_range"], "1-2")
        self.assertEqual(result["documents"]["beta.pdf"]["chunk_count"], 1)
        self.assertEqual(result["documents"]["beta.pdf"]["pages"], [5])
        self.assertEqual(result["documents"]["beta.pdf"]["page_range"], "5-5")


if __name__ == "__main__":
    unittest.main()
