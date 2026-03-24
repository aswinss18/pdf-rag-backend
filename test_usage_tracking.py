import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.auth import get_current_user, hash_password
from app.core.config import settings
from app.db.sqlite_store import create_user, get_usage_for_date, init_database, upsert_usage
from app.main import create_app
from app.services.usage_service import DAILY_REQUEST_LIMIT, get_today_string


class UsageTrackingTests(unittest.TestCase):
    def setUp(self):
        self._original_sqlite_db_path = settings.sqlite_db_path
        self._temp_dir = tempfile.TemporaryDirectory()
        settings.sqlite_db_path = os.path.join(self._temp_dir.name, "test.db")
        init_database()

        self.user = create_user("usage-tester", hash_password("password123"))
        self.app = create_app()
        self.app.dependency_overrides[get_current_user] = lambda: {
            "id": self.user["id"],
            "username": self.user["username"],
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        settings.sqlite_db_path = self._original_sqlite_db_path
        self._temp_dir.cleanup()

    def test_me_returns_usage_summary(self):
        response = self.client.get("/me")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["user"]["username"], self.user["username"])
        self.assertEqual(payload["usage"]["requests_used"], 0)
        self.assertEqual(payload["usage"]["requests_limit"], DAILY_REQUEST_LIMIT)
        self.assertEqual(payload["usage"]["tokens_used"], 0)

    def test_agent_tracks_request_and_tokens(self):
        with patch("app.api.routes.agent.run") as run_mock:
            run_mock.return_value = {
                "success": True,
                "answer": "Tracked response",
                "tools_used": 0,
                "tool_calls": [],
                "reasoning_steps": [],
                "memory_used": False,
                "tokens_used": 321,
            }

            response = self.client.post("/agent", json={"query": "Hello"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["usage"]["requests_used"], 1)
        self.assertEqual(payload["usage"]["requests_remaining"], DAILY_REQUEST_LIMIT - 1)
        self.assertEqual(payload["usage"]["tokens_used"], 321)

        usage = get_usage_for_date(self.user["id"], get_today_string())
        self.assertIsNotNone(usage)
        self.assertEqual(usage["requests"], 1)
        self.assertEqual(usage["tokens"], 321)

    def test_agent_blocks_after_daily_limit(self):
        upsert_usage(
            user_id=self.user["id"],
            date=get_today_string(),
            requests_delta=DAILY_REQUEST_LIMIT,
            tokens_delta=999,
        )

        response = self.client.post("/agent", json={"query": "One more"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "Daily limit reached")


if __name__ == "__main__":
    unittest.main()
