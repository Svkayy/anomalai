"""
Tests for supabase_database module — focused on the pure build_report_row function.
These tests do NOT require a live Supabase connection.
"""

import json
import pytest
from supabase_database import build_report_row


class TestBuildReportRow:
    """Tests for the pure build_report_row function."""

    def test_always_has_observations_key(self):
        """observations must be present regardless of input."""
        analysis = {
            "video_id": "vid_001",
            "video_duration": 30.0,
            "video_captured_at": "2026-06-23T12:00:00",
            "video_device_type": "smart glasses",
            "observations": [{"label": "exposed wiring", "severity": "high"}],
        }
        row = build_report_row(analysis)
        assert "observations" in row

    def test_observations_present_even_when_not_in_input(self):
        """observations defaults to empty list when not provided."""
        analysis = {
            "video_id": "vid_002",
            "video_duration": 15.5,
            "video_captured_at": "2026-06-23T12:00:00",
            "video_device_type": "smart glasses",
        }
        row = build_report_row(analysis)
        assert "observations" in row
        # Should be JSON-serialisable (list or dict)
        json.dumps(row["observations"])

    def test_observations_serializable_as_json(self):
        """observations value must be JSON-serialisable for jsonb storage."""
        analysis = {
            "video_id": "vid_003",
            "video_duration": 60.0,
            "video_captured_at": "2026-06-23T12:00:00",
            "video_device_type": "smart glasses",
            "observations": [{"label": "x", "severity": "low"}],
        }
        row = build_report_row(analysis)
        # Must not raise
        serialized = json.dumps(row["observations"])
        assert serialized is not None

    def test_video_id_in_row(self):
        """video_id from analysis is written into the row."""
        analysis = {
            "video_id": "my_video_42",
            "video_duration": 10.0,
            "video_captured_at": "2026-06-23T09:00:00",
            "video_device_type": "smart glasses",
        }
        row = build_report_row(analysis)
        assert row.get("video_id") == "my_video_42"

    def test_total_observations_defaults_to_zero(self):
        """total_observations defaults to 0 when not supplied."""
        analysis = {
            "video_id": "vid_004",
            "video_duration": 20.0,
            "video_captured_at": "2026-06-23T10:00:00",
            "video_device_type": "smart glasses",
        }
        row = build_report_row(analysis)
        assert row.get("total_observations") == 0

    def test_severity_counts_populated(self):
        """low/medium/high counts from analysis appear in the row."""
        analysis = {
            "video_id": "vid_005",
            "video_duration": 45.0,
            "video_captured_at": "2026-06-23T11:00:00",
            "video_device_type": "helmet-cam",
            "total_observations": 5,
            "low": 2,
            "medium": 2,
            "high": 1,
            "observations": [],
        }
        row = build_report_row(analysis)
        assert row["total_observations"] == 5
        assert row["low"] == 2
        assert row["medium"] == 2
        assert row["high"] == 1

    def test_no_description_column_in_row(self):
        """The old fallback description column must NOT appear in the row."""
        analysis = {
            "video_id": "vid_006",
            "video_duration": 5.0,
            "video_captured_at": "2026-06-23T08:00:00",
            "video_device_type": "smart glasses",
            "observations": [{"label": "slip hazard", "severity": "medium"}],
        }
        row = build_report_row(analysis)
        assert "description" not in row
