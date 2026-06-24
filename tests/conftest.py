"""
Shared pytest fixtures for Flask API tests.

The `client` fixture imports `app` and returns a test client.
No model initializers are called at import time (they live under
`if __name__ == '__main__':` in app.py), so no additional stubbing is needed.
"""

import pytest


@pytest.fixture
def client():
    """Return a Flask test client with TESTING=True."""
    import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
