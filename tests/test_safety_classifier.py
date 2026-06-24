"""
Tests for safety_classifier.py

Strategy:
- Unit-test the pure `is_dangerous(score, alpha)` helper — no model needed.
- Integration-test `classify_single_object` by monkeypatching the global
  instance's `.pipe` callable so no transformer model is loaded.
"""

import pytest
import safety_classifier as sc


# ---------------------------------------------------------------------------
# Pure threshold helper — fast, hermetic, no ML
# ---------------------------------------------------------------------------

class TestIsDangerous:
    """Tests for the module-level `is_dangerous(score, alpha)` helper."""

    def test_exactly_at_threshold_is_dangerous(self):
        # score == (1.0 - alpha)  → True (>= boundary)
        assert sc.is_dangerous(0.95, alpha=0.05) is True

    def test_above_threshold_is_dangerous(self):
        assert sc.is_dangerous(0.99, alpha=0.05) is True

    def test_below_threshold_is_safe(self):
        assert sc.is_dangerous(0.80, alpha=0.05) is False

    def test_custom_alpha_high_leniency(self):
        # alpha=0.5 → threshold=0.5; score 0.6 should be dangerous
        assert sc.is_dangerous(0.6, alpha=0.5) is True

    def test_custom_alpha_high_leniency_below(self):
        # alpha=0.5 → threshold=0.5; score 0.4 should be safe
        assert sc.is_dangerous(0.4, alpha=0.5) is False

    def test_zero_score_always_safe(self):
        assert sc.is_dangerous(0.0, alpha=0.05) is False

    def test_score_one_always_dangerous(self):
        assert sc.is_dangerous(1.0, alpha=0.05) is True


# ---------------------------------------------------------------------------
# classify_single_object — monkeypatched pipe, no model download
# ---------------------------------------------------------------------------

def _make_pipe_result(dangerous_score: float):
    """Return a callable that simulates the zero-shot pipeline output."""
    def fake_pipe(label, candidates):
        # zero-shot pipeline returns scores in the same order as candidates
        # candidates = ["dangerous", "safe"]
        safe_score = 1.0 - dangerous_score
        return {
            "labels": ["dangerous", "safe"],
            "scores": [dangerous_score, safe_score],
        }
    return fake_pipe


class TestClassifySingleObject:
    def test_dangerous_when_score_above_threshold(self, monkeypatch):
        # score=0.96 with default alpha=0.05 → threshold=0.95 → dangerous
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_pipe_result(0.96))
        monkeypatch.setattr(sc.safety_classifier, "initialized", True)

        result = sc.safety_classifier.classify_single_object("frayed live wire", alpha=0.05)

        assert result["is_dangerous"] is True
        assert result["is_safe"] is False
        assert result["classification"] == "dangerous"
        assert result["dangerous_score"] == pytest.approx(0.96, abs=0.001)

    def test_safe_when_score_below_threshold(self, monkeypatch):
        # score=0.2 with default alpha=0.05 → threshold=0.95 → safe
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_pipe_result(0.2))
        monkeypatch.setattr(sc.safety_classifier, "initialized", True)

        result = sc.safety_classifier.classify_single_object("tidy desk", alpha=0.05)

        assert result["is_dangerous"] is False
        assert result["is_safe"] is True
        assert result["classification"] == "safe"

    def test_result_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_pipe_result(0.5))
        monkeypatch.setattr(sc.safety_classifier, "initialized", True)

        result = sc.safety_classifier.classify_single_object("something", alpha=0.05)
        required_keys = {"is_dangerous", "is_safe", "dangerous_score", "safe_score",
                         "confidence", "classification"}
        assert required_keys.issubset(result.keys())

    def test_dangerous_score_and_safe_score_sum_to_one(self, monkeypatch):
        monkeypatch.setattr(sc.safety_classifier, "pipe", _make_pipe_result(0.7))
        monkeypatch.setattr(sc.safety_classifier, "initialized", True)

        result = sc.safety_classifier.classify_single_object("wet floor", alpha=0.05)
        assert result["dangerous_score"] + result["safe_score"] == pytest.approx(1.0, abs=0.01)
