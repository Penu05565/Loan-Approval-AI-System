# isort: skip_file
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app  # noqa: E402  (import must follow the sys.path fix-up above)


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
