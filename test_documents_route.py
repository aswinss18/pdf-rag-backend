import asyncio
import unittest

from app.api.routes.documents import list_documents
from app.db.vector_store import documents


class DocumentsRouteTests(unittest.TestCase):
    def setUp(self):
        self._original_documents = list(documents)
        documents.clear()

    def tearDown(self):
        documents.clear()
        documents.extend(self._original_documents)

    def test_list_documents_returns_summary_without_mutation_error(self):
        documents.extend(
            [
                {"doc": "alpha.pdf", "page": 1, "text": "a"},
                {"doc": "alpha.pdf", "page": 2, "text": "b"},
                {"doc": "alpha.pdf", "page": 2, "text": "c"},
                {"doc": "beta.pdf", "page": 5, "text": "d"},
            ]
        )

        result = asyncio.run(list_documents())

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
