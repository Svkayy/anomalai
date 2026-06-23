"""
Flask API / wiring tests — all models mocked via monkeypatch.

Contracts verified from app.py:

  GET /
    → 200, Content-Type: text/html

  POST /configure_parallel   (body: JSON with optional max_workers / batch_size /
                               target_points / max_masks)
    → 200, JSON {"success": True, "config": {...}}
       config keys: max_workers, batch_size, target_points, max_masks

  POST /api/classify_safety  (body: {"objects": [...], "alpha": float})
    → 400 when objects list is empty
    → 200 with JSON keys:
         classified_objects  (list, same length as input)
         total_objects       (int)
         dangerous_count     (int)
         safe_count          (int)
       and dangerous_count + safe_count == total_objects
"""

import json
import pytest
import safety_classifier as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_pipe(dangerous_score: float):
    """Return a callable that mimics the zero-shot pipeline for testing."""
    def fake_pipe(label, candidates):
        safe_score = 1.0 - dangerous_score
        return {
            "labels": list(candidates),
            "scores": [dangerous_score, safe_score],
        }
    return fake_pipe


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.content_type


# ---------------------------------------------------------------------------
# POST /configure_parallel
# ---------------------------------------------------------------------------

class TestConfigureParallel:
    def test_returns_200_with_valid_body(self, client):
        resp = client.post(
            "/configure_parallel",
            json={"max_workers": 8, "batch_size": 16},
        )
        assert resp.status_code == 200

    def test_response_shape(self, client):
        resp = client.post(
            "/configure_parallel",
            json={"max_workers": 4, "batch_size": 32, "target_points": 64, "max_masks": 25},
        )
        body = resp.get_json()
        assert body["success"] is True
        config = body["config"]
        for key in ("max_workers", "batch_size", "target_points", "max_masks"):
            assert key in config, f"missing config key: {key}"

    def test_custom_max_workers_is_reflected(self, client):
        resp = client.post("/configure_parallel", json={"max_workers": 12})
        body = resp.get_json()
        assert body["config"]["max_workers"] == 12

    def test_defaults_applied_when_keys_omitted(self, client):
        resp = client.post("/configure_parallel", json={})
        body = resp.get_json()
        config = body["config"]
        assert config["max_workers"] == 4
        assert config["batch_size"] == 32
        assert config["target_points"] == 64
        assert config["max_masks"] == 25


# ---------------------------------------------------------------------------
# POST /api/classify_safety
# ---------------------------------------------------------------------------

class TestClassifySafety:
    def test_empty_objects_returns_400(self, client, monkeypatch):
        # Even with a stubbed pipe, empty list triggers early 400
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.1))
        resp = client.post("/api/classify_safety", json={"objects": []})
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body

    def test_valid_objects_returns_200(self, client, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.1))
        objects = [{"label": "tidy desk", "coords": [0, 0, 100, 100]}]
        resp = client.post("/api/classify_safety", json={"objects": objects})
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.1))
        objects = [{"label": "hard hat", "coords": [0, 0, 50, 50]}]
        resp = client.post("/api/classify_safety", json={"objects": objects})
        body = resp.get_json()
        for key in ("classified_objects", "total_objects", "dangerous_count", "safe_count"):
            assert key in body, f"missing key: {key}"

    def test_counts_add_up_to_total(self, client, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.1))
        objects = [
            {"label": "hard hat", "coords": [0, 0, 50, 50]},
            {"label": "frayed wire", "coords": [10, 10, 60, 60]},
        ]
        resp = client.post("/api/classify_safety", json={"objects": objects})
        body = resp.get_json()
        assert body["total_objects"] == len(objects)
        assert body["dangerous_count"] + body["safe_count"] == body["total_objects"]

    def test_classified_objects_length_matches_input(self, client, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.2))
        objects = [
            {"label": "a", "coords": [0, 0, 10, 10]},
            {"label": "b", "coords": [0, 0, 10, 10]},
            {"label": "c", "coords": [0, 0, 10, 10]},
        ]
        resp = client.post("/api/classify_safety", json={"objects": objects})
        body = resp.get_json()
        assert len(body["classified_objects"]) == 3

    def test_dangerous_object_counted_when_score_high(self, client, monkeypatch):
        # dangerous_score=0.97 with default alpha=0.05 → threshold 0.95 → dangerous
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.97))
        objects = [{"label": "live wire", "coords": [0, 0, 20, 20]}]
        resp = client.post(
            "/api/classify_safety",
            json={"objects": objects, "alpha": 0.05},
        )
        body = resp.get_json()
        assert body["dangerous_count"] == 1
        assert body["safe_count"] == 0
        classified = body["classified_objects"][0]
        assert classified["safety"]["is_dangerous"] is True
        assert classified["safety"]["classification"] == "dangerous"

    def test_safe_object_when_score_low(self, client, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_fake_pipe(0.10))
        objects = [{"label": "coffee mug", "coords": [0, 0, 20, 20]}]
        resp = client.post(
            "/api/classify_safety",
            json={"objects": objects, "alpha": 0.05},
        )
        body = resp.get_json()
        assert body["dangerous_count"] == 0
        assert body["safe_count"] == 1
        classified = body["classified_objects"][0]
        assert classified["safety"]["is_dangerous"] is False
        assert classified["safety"]["classification"] == "safe"
