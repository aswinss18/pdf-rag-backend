"""
Application entrypoint.
This file is a thin wrapper — all logic lives inside app/.
Run with: uvicorn main:app --reload
"""

import os
import uvicorn
from app.main import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)