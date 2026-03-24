import unittest
from datetime import datetime, timedelta, timezone

from app.core import auth
from app.core.auth import JWTError, create_access_token, hash_password, verify_password
from app.core.config import settings


class AuthTokenTests(unittest.TestCase):
    def setUp(self):
        self.original_secret = settings.jwt_secret_key
        self.original_algorithm = settings.jwt_algorithm
        self.original_expiry = settings.access_token_expire_minutes
        settings.jwt_secret_key = "test-secret"
        settings.jwt_algorithm = "HS256"
        settings.access_token_expire_minutes = 5

    def tearDown(self):
        settings.jwt_secret_key = self.original_secret
        settings.jwt_algorithm = self.original_algorithm
        settings.access_token_expire_minutes = self.original_expiry

    def test_create_and_decode_access_token(self):
        token = create_access_token({"sub": "tester", "user_id": 123})

        payload = auth._decode_jwt(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

        self.assertEqual(payload["sub"], "tester")
        self.assertEqual(payload["user_id"], 123)
        self.assertIn("exp", payload)

    def test_decode_rejects_expired_token(self):
        expired_payload = {
            "sub": "tester",
            "user_id": 123,
            "exp": int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp()),
        }
        token = auth._encode_jwt(expired_payload, settings.jwt_secret_key, settings.jwt_algorithm)

        with self.assertRaises(JWTError):
            auth._decode_jwt(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

    def test_hash_and_verify_password(self):
        password_hash = hash_password("s3cret-pass")

        self.assertNotEqual(password_hash, "s3cret-pass")
        self.assertTrue(verify_password("s3cret-pass", password_hash))
        self.assertFalse(verify_password("wrong-pass", password_hash))


if __name__ == "__main__":
    unittest.main()
