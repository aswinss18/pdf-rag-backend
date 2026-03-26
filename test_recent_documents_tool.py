import unittest

from app.tools import document_tools


class RecentDocumentsToolTests(unittest.TestCase):
    def test_list_recent_documents_orders_by_uploaded_at_desc(self):
        original_get_current_user_id = document_tools.get_current_user_id
        original_get_documents = document_tools.get_documents

        try:
            document_tools.get_current_user_id = lambda: 7
            document_tools.get_documents = lambda user_id: [
                {
                    "doc": "older.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "text": "older",
                    "created_at": "2026-03-26T10:00:00+00:00",
                },
                {
                    "doc": "newer.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "text": "newer",
                    "created_at": "2026-03-26T11:00:00+00:00",
                },
            ]

            result = document_tools.list_recent_documents(limit=2)

            self.assertTrue(result["success"])
            self.assertEqual(result["documents"][0]["filename"], "newer.pdf")
            self.assertEqual(result["documents"][1]["filename"], "older.pdf")
        finally:
            document_tools.get_current_user_id = original_get_current_user_id
            document_tools.get_documents = original_get_documents


if __name__ == "__main__":
    unittest.main()
