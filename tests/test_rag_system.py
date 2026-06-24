"""
Tests for rag_system.py — fallback path (RAG_AVAILABLE = False).

All tests monkeypatch RAG_AVAILABLE so no Supabase/Gemini/Ollama calls are made.
"""

import pytest
import rag_system


# ---------------------------------------------------------------------------
# is_rag_available
# ---------------------------------------------------------------------------

class TestIsRagAvailable:
    def test_returns_false_when_rag_unavailable(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False)
        assert rag_system.is_rag_available() is False

    def test_returns_true_when_rag_available(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", True)
        assert rag_system.is_rag_available() is True


# ---------------------------------------------------------------------------
# generate_formal_safety_report — fallback string
# ---------------------------------------------------------------------------

SAMPLE_OBSERVATIONS = {
    "video_analysis_summary": {
        "total_frames_analyzed": 10,
        "unsafe_frames_count": 3,
    },
    "frame_analyses": [
        {
            "frame_id": 1,
            "observations": [
                {
                    "label": "exposed wiring",
                    "severity": "high",
                    "reasons": ["electrical hazard", "no insulation"],
                }
            ],
        }
    ],
}


class TestGenerateFormalSafetyReportFallback:
    def test_fallback_returns_non_empty_string(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False)
        report = rag_system.generate_formal_safety_report(SAMPLE_OBSERVATIONS)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_fallback_string_indicates_unavailability(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False)
        report = rag_system.generate_formal_safety_report(SAMPLE_OBSERVATIONS)
        # The real fallback string is: "RAG system not available - cannot generate formal report"
        assert "not available" in report.lower()

    def test_fallback_works_with_empty_observations(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False)
        report = rag_system.generate_formal_safety_report({})
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# build_context — fallback returns empty string
# ---------------------------------------------------------------------------

class TestBuildContextFallback:
    def test_returns_empty_string_when_rag_unavailable(self, monkeypatch):
        monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False)
        ctx = rag_system.build_context("exposed wiring")
        assert ctx == ""
