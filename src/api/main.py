"""
FastAPI entry point.

Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from src.api import app  # noqa: F401
