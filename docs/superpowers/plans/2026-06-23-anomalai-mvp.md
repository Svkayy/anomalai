# anomalai MVP Buildout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `anomalai` to a portfolio-credible MVP: the CV pipeline runs cold on macOS, every external integration is coherently wired, there is a pytest suite, and a clean README with diagrams + captured demo proof.

**Architecture:** Flask web app. A SAM2 (CoreML) → CLIP → BART-safety → Gemini → RAG/OSHA → Supabase pipeline. Pure helpers get extracted from the `app.py` monolith into `geometry.py` and `vocabulary.py` so they are unit-testable; external services are mocked in tests (no live keys).

**Tech Stack:** Python 3.11, Flask, coremltools (SAM2), PyTorch + HuggingFace transformers (CLIP, BART, Depth-Anything), Google Gemini SDK, Ollama + Supabase/pgvector (RAG + storage), pytest.

## Global Constraints

- **Python 3.11** exactly (MediaPipe/CoreML/transformers compatibility floor).
- **Platform: macOS** — CoreML is the segmentation backend; do not attempt cross-OS support beyond a non-crashing font fallback.
- **No live external calls in tests** — SAM2, CLIP, Gemini, Ollama, and Supabase are always mocked. No secrets in the repo.
- **Close holes, don't camouflage** — complete real code (schema, integrations); never fabricate functionality or demo output.
- **Model weights are never committed** — fetched via `scripts/download_models.py` into `models/` (git-ignored).
- **Working branch:** `mvp-buildout`. Commit after every task.
- Source of weights: HuggingFace `apple/coreml-sam2.1-small` (3 FLOAT16 `.mlpackage` files).
- Expected model filenames (referenced in `app.py:76-79`, `script.py:427-429`):
  `SAM2_1SmallImageEncoderFLOAT16.mlpackage`, `SAM2_1SmallPromptEncoderFLOAT16.mlpackage`, `SAM2_1SmallMaskDecoderFLOAT16.mlpackage`.

---

## File Structure

- `scripts/download_models.py` *(new)* — fetch + rename SAM2 CoreML weights.
- `.env.example` *(new)* — documented config template.
- `settings.py` *(new)* — central env/config loader (single source of truth).
- `geometry.py` *(new)* — pure mask/box geometry: `mask_to_box`, `box_area`, `box_iou`, `mask_iou`, `remove_small_regions`.
- `vocabulary.py` *(new)* — `generate_workplace_vocabulary`.
- `setup_database.sql` *(modify)* — complete schema.
- `supabase_database.py` *(modify)* — remove runtime column-existence fallback.
- `video_processor.py` *(modify)* — fix frame-interval mismatch, make configurable.
- `script.py` *(modify)* — `MODEL_SIZE` switch instead of commented-out paths.
- `app.py` *(modify)* — import from `geometry`/`vocabulary`/`settings`; font fallback; bbox validation; `logging` instead of `print`.
- `tests/` *(new)* — `test_geometry.py`, `test_vocabulary.py`, `test_safety_classifier.py`, `test_rag_system.py`, `test_supabase_database.py`, `test_api.py`, `conftest.py`.
- `pytest.ini` *(new)*, `.github/workflows/tests.yml` *(new)*.
- `README.md` *(modify)* + `docs/diagrams/` *(new)* + `docs/demo/` *(new)*.

---

## Task 1: Model download script

**Files:**
- Create: `scripts/download_models.py`
- Modify: `.gitignore` (ensure `models/` weights ignored)
- Test: `tests/test_download_models.py`

**Interfaces:**
- Produces: `resolve_target_names(repo_files: list[str]) -> dict[str, str]` mapping a downloaded HF filename → the expected local filename; `EXPECTED_FILES: list[str]` (the 3 names from Global Constraints).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_download_models.py
from scripts.download_models import EXPECTED_FILES, resolve_target_names

def test_expected_files_are_the_three_mlpackages():
    assert EXPECTED_FILES == [
        "SAM2_1SmallImageEncoderFLOAT16.mlpackage",
        "SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
        "SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
    ]

def test_resolve_maps_encoder_decoder_by_keyword():
    repo_files = [
        "SAM2_1SmallImageEncoderFLOAT16.mlpackage",
        "SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
        "SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
    ]
    mapping = resolve_target_names(repo_files)
    assert mapping["SAM2_1SmallImageEncoderFLOAT16.mlpackage"] == "SAM2_1SmallImageEncoderFLOAT16.mlpackage"
    assert set(mapping.values()) == set(EXPECTED_FILES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_download_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.download_models'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/download_models.py
"""Download Apple's CoreML SAM2.1-small weights into ./models.

Usage: python scripts/download_models.py
Requires: pip install huggingface_hub
"""
from __future__ import annotations
import os
import shutil

HF_REPO = "apple/coreml-sam2.1-small"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

EXPECTED_FILES = [
    "SAM2_1SmallImageEncoderFLOAT16.mlpackage",
    "SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
    "SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
]

_ROLE_KEYWORDS = {
    "ImageEncoder": "SAM2_1SmallImageEncoderFLOAT16.mlpackage",
    "PromptEncoder": "SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
    "MaskDecoder": "SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
}

def resolve_target_names(repo_files: list[str]) -> dict[str, str]:
    """Map each downloaded .mlpackage to the local filename the app expects."""
    mapping: dict[str, str] = {}
    for fname in repo_files:
        if not fname.endswith(".mlpackage"):
            continue
        for keyword, target in _ROLE_KEYWORDS.items():
            if keyword.lower() in fname.lower():
                mapping[fname] = target
                break
    return mapping

def main() -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(MODELS_DIR, exist_ok=True)
    local = snapshot_download(repo_id=HF_REPO, allow_patterns=["*.mlpackage*"])
    repo_files = os.listdir(local)
    mapping = resolve_target_names(repo_files)
    if set(mapping.values()) != set(EXPECTED_FILES):
        raise SystemExit(
            f"Could not resolve all expected models. Found: {sorted(mapping.values())}"
        )
    for src_name, target_name in mapping.items():
        src = os.path.join(local, src_name)
        dst = os.path.join(MODELS_DIR, target_name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Installed {target_name}")
    print(f"Done. Models in {MODELS_DIR}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_download_models.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Ensure weights are git-ignored**

Confirm `.gitignore` contains `models/` (add the line if absent). The empty `models/.gitkeep` stays tracked.

- [ ] **Step 6: Commit**

```bash
git add scripts/download_models.py tests/test_download_models.py .gitignore
git commit -m "feat: add SAM2 CoreML model download script"
```

---

## Task 2: Central config + .env.example

**Files:**
- Create: `settings.py`, `.env.example`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `get_settings() -> Settings` where `Settings` has attributes `supabase_url, supabase_key, gemini_api_key, ollama_host, model_size, frame_interval` (all read from env with documented defaults). `Settings.rag_enabled -> bool` (True only when supabase + ollama configured).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings.py
import importlib

def test_defaults_when_env_absent(monkeypatch):
    for k in ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY", "OLLAMA_HOST", "MODEL_SIZE", "FRAME_INTERVAL"]:
        monkeypatch.delenv(k, raising=False)
    import settings as s
    importlib.reload(s)
    cfg = s.get_settings()
    assert cfg.model_size == "small"
    assert cfg.frame_interval == 4
    assert cfg.ollama_host == "http://localhost:11434"
    assert cfg.rag_enabled is False

def test_rag_enabled_when_supabase_and_ollama_present(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    import settings as s
    importlib.reload(s)
    cfg = s.get_settings()
    assert cfg.rag_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'settings'`

- [ ] **Step 3: Write minimal implementation**

```python
# settings.py
from __future__ import annotations
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@dataclass(frozen=True)
class Settings:
    supabase_url: str | None
    supabase_key: str | None
    gemini_api_key: str | None
    ollama_host: str
    model_size: str
    frame_interval: int

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def rag_enabled(self) -> bool:
        return self.supabase_enabled and bool(self.ollama_host)

def get_settings() -> Settings:
    return Settings(
        supabase_url=os.getenv("SUPABASE_URL") or None,
        supabase_key=os.getenv("SUPABASE_KEY") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model_size=os.getenv("MODEL_SIZE", "small"),
        frame_interval=int(os.getenv("FRAME_INTERVAL", "4")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_settings.py -v`
Expected: PASS

- [ ] **Step 5: Write `.env.example`**

```bash
# .env.example — copy to .env and fill in. None are required to run the
# local CV pipeline; they enable the cloud/LLM features.

# Supabase (report storage + pgvector RAG index)
SUPABASE_URL=
SUPABASE_KEY=

# Google Gemini (hazard analysis + report narrative)
GEMINI_API_KEY=

# Ollama host for RAG embeddings (nomic-embed-text). Default shown.
OLLAMA_HOST=http://localhost:11434

# SAM2 model size: small (default) or large
MODEL_SIZE=small

# Video sampling: analyze every Nth frame
FRAME_INTERVAL=4
```

- [ ] **Step 6: Commit**

```bash
git add settings.py .env.example tests/test_settings.py
git commit -m "feat: add central settings loader and .env.example"
```

---

## Task 3: Complete the database schema; drop the runtime column fallback

**Files:**
- Modify: `setup_database.sql`
- Modify: `supabase_database.py` (remove the "does `observations` column exist?" runtime branch, ~lines 45-58, 99-101)
- Test: `tests/test_supabase_database.py`

**Interfaces:**
- Consumes: `settings.get_settings()`.
- Produces: `build_report_row(analysis: dict) -> dict` — a pure function that builds the row dict written to the `reports` table, always including an `observations` column.

- [ ] **Step 1: Read the current persistence code**

Read `supabase_database.py` in full. Identify the insert/update payload construction and the conditional that checks whether `observations` exists, falling back to `description`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_supabase_database.py
from supabase_database import build_report_row

def test_build_report_row_always_has_observations():
    analysis = {"observations": [{"label": "exposed wiring", "danger": True}],
                "image_id": "abc"}
    row = build_report_row(analysis)
    assert "observations" in row
    assert row["image_id"] == "abc"

def test_build_report_row_serializes_observations_as_json():
    import json
    analysis = {"observations": [{"label": "x"}], "image_id": "i"}
    row = build_report_row(analysis)
    # observations stored as JSON-serializable structure
    json.dumps(row["observations"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_supabase_database.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_report_row'`

- [ ] **Step 4: Implement `build_report_row` and remove the fallback**

Extract payload construction into a pure `build_report_row(analysis: dict) -> dict` at module top of `supabase_database.py` that always sets `observations`. Replace the conditional column logic in the insert/update methods with a direct call to `build_report_row`. Delete the runtime column-existence check.

```python
# supabase_database.py (add near top, then call from insert/update)
def build_report_row(analysis: dict) -> dict:
    return {
        "image_id": analysis.get("image_id"),
        "observations": analysis.get("observations", []),
        "summary": analysis.get("summary", ""),
    }
```

(Preserve any other columns the existing insert used; merge them into the dict.)

- [ ] **Step 5: Complete `setup_database.sql`**

Ensure full `CREATE TABLE IF NOT EXISTS` statements exist for every table the code touches, with the `observations` column present:

```sql
-- setup_database.sql
create extension if not exists vector;

create table if not exists reports (
    id uuid primary key default gen_random_uuid(),
    image_id text,
    observations jsonb not null default '[]'::jsonb,
    summary text default '',
    created_at timestamptz default now()
);

create table if not exists formal_reports (
    id uuid primary key default gen_random_uuid(),
    report_id uuid references reports(id),
    title text,
    body text,
    osha_references jsonb default '[]'::jsonb,
    created_at timestamptz default now()
);

create table if not exists policy_embeddings (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    embedding vector(768),
    source text,
    created_at timestamptz default now()
);
```

Reconcile column names with the actual reads/writes in `supabase_database.py` and `rag_system.py`; adjust the SQL so they match exactly.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_supabase_database.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add setup_database.sql supabase_database.py tests/test_supabase_database.py
git commit -m "feat: complete DB schema and remove runtime column fallback"
```

---

## Task 4: Extract pure geometry helpers into `geometry.py`

**Files:**
- Create: `geometry.py`
- Modify: `app.py` (remove the moved defs at lines ~137-167, ~558-585; add `from geometry import mask_to_box, box_area, box_iou, mask_iou, remove_small_regions`)
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `mask_to_box(mask)`, `box_area(box)`, `box_iou(a, b)`, `mask_iou(a_mask, b_mask)`, `remove_small_regions(mask, min_region_area=0)` — moved verbatim from `app.py`, behavior unchanged.

- [ ] **Step 1: Read the current implementations**

Read `app.py:137-167` (`mask_to_box`, `box_area`, `box_iou`, `mask_iou`) and `app.py:558-585` (`remove_small_regions`). Copy their bodies exactly.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_geometry.py
import numpy as np
from geometry import mask_to_box, box_area, box_iou, mask_iou

def _mask(x0, y0, x1, y1, h=10, w=10):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m

def test_box_area():
    assert box_area([0, 0, 2, 3]) == 6

def test_box_iou_identical_is_one():
    assert box_iou([0, 0, 4, 4], [0, 0, 4, 4]) == 1.0

def test_box_iou_disjoint_is_zero():
    assert box_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0

def test_mask_to_box_tight_bounds():
    box = mask_to_box(_mask(2, 3, 6, 7))
    assert list(box) == [2, 3, 6, 7]

def test_mask_iou_identical_is_one():
    m = _mask(0, 0, 5, 5)
    assert mask_iou(m, m) == 1.0

def test_mask_iou_disjoint_is_zero():
    assert mask_iou(_mask(0, 0, 2, 2), _mask(5, 5, 8, 8)) == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geometry'`

- [ ] **Step 4: Create `geometry.py` and update `app.py`**

Move the five functions verbatim into `geometry.py` (add `import numpy as np`, and `import cv2` if `remove_small_regions` uses it). In `app.py`, delete the moved defs and add at the top with the other imports:

```python
from geometry import mask_to_box, box_area, box_iou, mask_iou, remove_small_regions
```

If `mask_to_box` returns a numpy array, adapt the test assertions to `list(box) == [...]` (already done above).

- [ ] **Step 5: Run tests + import-smoke**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS
Run: `python -c "import app"` is **not** required here (app imports heavy models). Instead: `python -c "import geometry"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add geometry.py app.py tests/test_geometry.py
git commit -m "refactor: extract pure geometry helpers into geometry.py"
```

---

## Task 5: Extract `generate_workplace_vocabulary` into `vocabulary.py`

**Files:**
- Create: `vocabulary.py`
- Modify: `app.py` (remove def at ~430-466; add `from vocabulary import generate_workplace_vocabulary`)
- Test: `tests/test_vocabulary.py`

**Interfaces:**
- Produces: `generate_workplace_vocabulary() -> list[str]` — moved verbatim.

- [ ] **Step 1: Read the current implementation**

Read `app.py:430-466`. Copy the returned vocabulary list exactly.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vocabulary.py
from vocabulary import generate_workplace_vocabulary

def test_returns_nonempty_unique_strings():
    vocab = generate_workplace_vocabulary()
    assert isinstance(vocab, list)
    assert len(vocab) >= 20
    assert all(isinstance(v, str) and v for v in vocab)
    assert len(vocab) == len(set(vocab))  # no duplicates
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vocabulary'`

- [ ] **Step 4: Create `vocabulary.py` and update `app.py`**

Move `generate_workplace_vocabulary` verbatim into `vocabulary.py`. In `app.py` delete the def and add `from vocabulary import generate_workplace_vocabulary`. If the test's uniqueness assertion fails because the source list has duplicates, de-duplicate the list in `vocabulary.py` (a legitimate hole-closing fix).

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add vocabulary.py app.py tests/test_vocabulary.py
git commit -m "refactor: extract workplace vocabulary into vocabulary.py"
```

---

## Task 6: Close holes — frame interval, font fallback, bbox validation, logging, MODEL_SIZE

**Files:**
- Modify: `video_processor.py` (frame interval), `app.py` (font, bbox validation, logging), `script.py` (MODEL_SIZE switch)
- Test: `tests/test_video_processor.py`, add a case to `tests/test_api.py` later (Task 9)

**Interfaces:**
- Produces: `get_font(size: int)` in `app.py` returning a usable `PIL.ImageFont` on any platform; `validate_bbox(coords, width, height) -> tuple[int,int,int,int]` raising `ValueError` on invalid input; `extract_frames(..., frame_interval: int | None = None)` honoring `settings.frame_interval`.

- [ ] **Step 1: Write the failing test (frame interval)**

```python
# tests/test_video_processor.py
import inspect
import video_processor

def test_extract_frames_accepts_frame_interval_param():
    sig = inspect.signature(video_processor.extract_frames)
    assert "frame_interval" in sig.parameters
```

(If the function has a different name, adjust to the real frame-extraction entry point in `video_processor.py`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_video_processor.py -v`
Expected: FAIL — no `frame_interval` parameter.

- [ ] **Step 3: Fix the frame-interval mismatch + make configurable**

In `video_processor.py`, locate `frame_interval = 4  # ... 7 ...` (line ~48). Replace the hardcoded constant + misleading comment with a parameter:

```python
from settings import get_settings

def extract_frames(video_path, output_dir, frame_interval: int | None = None):
    if frame_interval is None:
        frame_interval = get_settings().frame_interval
    # ... existing body, using frame_interval ...
```

- [ ] **Step 4: Add cross-platform font helper**

In `app.py`, replace the two hardcoded `ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)` calls (lines ~1225, ~1433) with a shared helper:

```python
from PIL import ImageFont

def get_font(size: int):
    for path in ("/System/Library/Fonts/Arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
```

- [ ] **Step 5: Add bbox validation**

In `app.py` add and call before segmentation in `/segment` (around lines 940-949):

```python
def validate_bbox(coords, width, height):
    try:
        x0, y0, x1, y1 = (int(round(float(c))) for c in coords)
    except (TypeError, ValueError):
        raise ValueError("bbox must be four numbers")
    x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
    y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox has zero area after clamping")
    return x0, y0, x1, y1
```

- [ ] **Step 6: Swap `print()` for `logging`**

At the top of `app.py` add:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("anomalai")
```

Replace `print(...)` diagnostic calls in `app.py` with `logger.info(...)` / `logger.warning(...)` / `logger.error(...)` as appropriate. (Leave any intentional user-facing CLI prints.)

- [ ] **Step 7: MODEL_SIZE switch in `script.py`**

Replace the commented-out Large paths (`script.py:424-429`) with a clean switch driven by settings:

```python
from settings import get_settings

_MODEL_FILES = {
    "small": ("SAM2_1SmallImageEncoderFLOAT16.mlpackage",
              "SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
              "SAM2_1SmallMaskDecoderFLOAT16.mlpackage"),
    "large": ("SAM2_1LargeImageEncoderFLOAT16.mlpackage",
              "SAM2_1LargePromptEncoderFLOAT16.mlpackage",
              "SAM2_1LargeMaskDecoderFLOAT16.mlpackage"),
}

def model_paths(models_dir="./models", size=None):
    size = size or get_settings().model_size
    enc, prompt, dec = _MODEL_FILES[size]
    import os
    return (os.path.join(models_dir, enc),
            os.path.join(models_dir, prompt),
            os.path.join(models_dir, dec))
```

Update the `sam.load_models(...)` call(s) in `script.py` and `app.py:76-79` to use `model_paths()`.

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_video_processor.py -v`
Expected: PASS
Run: `python -c "import geometry, vocabulary, settings, video_processor"`
Expected: no error.

- [ ] **Step 9: Commit**

```bash
git add video_processor.py app.py script.py tests/test_video_processor.py
git commit -m "fix: close holes (frame interval, fonts, bbox validation, logging, model-size switch)"
```

---

## Task 7: Tests for safety classifier + RAG fallback (mocked)

**Files:**
- Test: `tests/test_safety_classifier.py`, `tests/test_rag_system.py`
- Modify (only if needed for testability): `safety_classifier.py`, `rag_system.py`

**Interfaces:**
- Consumes: `safety_classifier` decision logic (threshold at `safety_classifier.py:74`), `rag_system` fallback path (`rag_system.py:109-121`).

- [ ] **Step 1: Read both modules**

Read `safety_classifier.py` and `rag_system.py`. Identify (a) the pure decision function that turns a `dangerous_score` into a safe/dangerous verdict, and (b) the fallback that returns hardcoded OSHA references when RAG deps are unavailable.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_safety_classifier.py
import safety_classifier as sc

def test_dangerous_above_threshold(monkeypatch):
    # Mock the zero-shot pipeline so no model download happens.
    monkeypatch.setattr(sc, "_classifier", lambda *a, **k: {
        "labels": ["dangerous", "safe"], "scores": [0.9, 0.1]}, raising=False)
    verdict = sc.classify("a frayed live wire", alpha=0.5)
    assert verdict["dangerous"] is True

def test_safe_below_threshold(monkeypatch):
    monkeypatch.setattr(sc, "_classifier", lambda *a, **k: {
        "labels": ["dangerous", "safe"], "scores": [0.2, 0.8]}, raising=False)
    verdict = sc.classify("a tidy desk", alpha=0.5)
    assert verdict["dangerous"] is False
```

```python
# tests/test_rag_system.py
import rag_system

def test_fallback_returns_osha_references_when_rag_unavailable(monkeypatch):
    monkeypatch.setattr(rag_system, "RAG_AVAILABLE", False, raising=False)
    refs = rag_system.get_osha_references("exposed wiring")
    assert isinstance(refs, list)
    assert len(refs) >= 1
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_safety_classifier.py tests/test_rag_system.py -v`
Expected: FAIL — names/signatures don't yet match.

- [ ] **Step 4: Adapt modules minimally to expose testable seams**

If `safety_classifier.py` does classification inline, extract a `classify(text, alpha=...) -> dict` function (and a module-level `_classifier` the test can monkeypatch) preserving the existing `dangerous_score >= (1.0 - alpha)` logic. If `rag_system.py`'s fallback is inline, expose `get_osha_references(query) -> list` that returns the hardcoded references when `RAG_AVAILABLE` is False. Keep behavior identical; only restructure for a seam.

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/test_safety_classifier.py tests/test_rag_system.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add safety_classifier.py rag_system.py tests/test_safety_classifier.py tests/test_rag_system.py
git commit -m "test: cover safety threshold and RAG fallback with mocks"
```

---

## Task 8: Flask API/wiring tests with all models mocked

**Files:**
- Create: `tests/conftest.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: the Flask `app` object and its routes (`/`, `/api/classify_safety`, `/configure_parallel`).
- Produces: a `client` pytest fixture with SAM2/CLIP/Gemini initializers patched to no-ops so importing `app` does not download or load models.

- [ ] **Step 1: Write `conftest.py` that neutralizes heavy init**

```python
# tests/conftest.py
import sys
import types
import pytest

@pytest.fixture
def client(monkeypatch):
    # Stub the four initializers BEFORE importing app so no models load.
    import importlib
    monkeypatch.setenv("MODEL_SIZE", "small")
    app_module = importlib.import_module("app")
    for name in ("initialize_sam", "initialize_depth", "initialize_clip", "initialize_gemini"):
        if hasattr(app_module, name):
            monkeypatch.setattr(app_module, name, lambda *a, **k: None)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()
```

(If importing `app` triggers model loading at import time, move the init calls in `app.py` under `if __name__ == "__main__":` so import is side-effect-free. That is itself a hole-closing improvement — make this change if needed.)

- [ ] **Step 2: Write the failing API tests**

```python
# tests/test_api.py
def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_configure_parallel_accepts_config(client):
    resp = client.post("/configure_parallel", json={"max_workers": 8})
    assert resp.status_code in (200, 400)  # endpoint exists and is wired

def test_classify_safety_endpoint_wired(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "classify_safety_text",
                        lambda *a, **k: {"dangerous": True}, raising=False)
    resp = client.post("/api/classify_safety", json={"label": "live wire"})
    assert resp.status_code in (200, 400, 422)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL initially (import side effects or route mismatch).

- [ ] **Step 4: Make `app` import side-effect-free**

Ensure the model `initialize_*()` calls and `app.run(...)` live under `if __name__ == "__main__":` (near `app.py:2971`). Adjust route handler names referenced in the test to the actual ones if different.

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_api.py app.py
git commit -m "test: Flask API/wiring tests with models mocked; make app import side-effect-free"
```

---

## Task 9: pytest config + CI workflow + model-gated smoke test

**Files:**
- Create: `pytest.ini`, `.github/workflows/tests.yml`, `tests/test_smoke.py`

- [ ] **Step 1: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    smoke: real-pipeline test, requires downloaded SAM2 weights (skipped by default)
```

- [ ] **Step 2: Write the model-gated smoke test**

```python
# tests/test_smoke.py
import os
import pytest

MODELS = "./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage"

@pytest.mark.smoke
@pytest.mark.skipif(not os.path.exists(MODELS), reason="SAM2 weights not downloaded")
def test_sam_loads():
    from script import SAM2  # adjust to the real class name
    sam = SAM2()
    sam.load_models(*__import__("script").model_paths())
    assert sam is not None
```

- [ ] **Step 3: Run the full non-smoke suite**

Run: `python -m pytest -m "not smoke" -v`
Expected: PASS (all prior tests green; smoke skipped).

- [ ] **Step 4: Write CI workflow (non-model tests only)**

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest -m "not smoke" -v
```

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/test_smoke.py .github/workflows/tests.yml
git commit -m "test: add pytest config, model-gated smoke test, and CI workflow"
```

---

## Task 10: Baseline run + capture demo proof (manual checkpoint)

**Files:**
- Create: `docs/demo/` (screenshots, short screen recording, sample annotated outputs)

> **Stop and coordinate with the user here.** This is the point to (optionally) drop a temporary `GEMINI_API_KEY` into `.env` for live report screenshots. Without it, the Gemini/RAG panels render via the built-in fallback.

- [ ] **Step 1: Set up the environment**

```bash
cd /Users/sandeepvinay.sk/CMU/anomalai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
cp .env.example .env   # fill in keys only if capturing live LLM output
```

- [ ] **Step 2: Run the app**

Run: `python app.py`
Expected: Flask serves on `http://localhost:5004`; startup logs show SAM2, Depth, CLIP initialized.

- [ ] **Step 3: Exercise the pipeline in the browser**

Upload a sample workplace image → run segmentation → confirm masks, CLIP labels, and safety flags render. Repeat with a short video. Generate a formal report.

- [ ] **Step 4: Capture proof**

Save annotated output images, 3–5 UI screenshots, and a ~20s screen recording into `docs/demo/`. Name them descriptively (`segmentation.png`, `safety-flags.png`, `formal-report.png`, `video-analysis.png`, `demo.mp4`).

- [ ] **Step 5: Commit (proof only; no weights, no .env)**

```bash
git add docs/demo
git commit -m "docs: add captured demo proof"
```

---

## Task 11: README rewrite + architecture & tech-stack diagrams

**Files:**
- Modify: `README.md`
- Create: `docs/diagrams/architecture.svg`, `docs/diagrams/tech-stack.svg` (+ editable `.excalidraw` sources)

- [ ] **Step 1: Generate the two diagrams**

Use the `/diagram` skill (or mermaid → SVG) to produce:
1. **Architecture/pipeline:** `image|video → SAM2 (parallel grid) → CLIP labels → BART safety → Gemini hazards → RAG/OSHA report → Supabase`.
2. **Tech stack:** grouped by layer — Web (Flask, HTML/JS), CV/ML (CoreML SAM2, CLIP, Depth-Anything, BART), LLM/RAG (Gemini, Ollama nomic-embed, pgvector), Data (Supabase).

Save SVGs + editable sources under `docs/diagrams/`.

- [ ] **Step 2: Rewrite `README.md`**

Sections in order: one-line tagline; demo GIF/screenshot (from `docs/demo/`); **Architecture** (embed `architecture.svg`); **Tech Stack** (embed `tech-stack.svg`); **Highlight: parallel grid segmentation** (link `PARALLEL_PROCESSING.md`); **Setup** (`python scripts/download_models.py`, `.env`, `setup_database.sql`); **Usage**; **Testing** (`pytest -m "not smoke"`); **Project structure**; links to deep-dive docs (`VIDEO_ANALYSIS.md`, `LABELING_GUIDE.md`).

- [ ] **Step 3: Verify links + images resolve**

Run: `python -c "import re,sys; [sys.exit('missing '+p) for p in re.findall(r'\]\((docs/[^)]+)\)', open('README.md').read()) if not __import__('os').path.exists(p)]"`
Expected: no output (all referenced doc/image paths exist).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/diagrams
git commit -m "docs: rewrite README with architecture + tech-stack diagrams"
```

---

## Task 12: Final verification + push

- [ ] **Step 1: Run the full non-smoke suite**

Run: `python -m pytest -m "not smoke" -v`
Expected: all PASS.

- [ ] **Step 2: Import-smoke every new/refactored module**

Run: `python -c "import settings, geometry, vocabulary, video_processor, safety_classifier, rag_system, supabase_database"`
Expected: no error.

- [ ] **Step 3: Confirm no stray TODO/print debug remains**

Run: `grep -rnE "TODO|FIXME|XXX" --include=*.py . | grep -v tests/`
Expected: empty (or only intentional, documented items).

- [ ] **Step 4: Push the branch**

```bash
git push -u origin mvp-buildout
```

- [ ] **Step 5: Open a PR (optional)**

```bash
gh pr create --title "anomalai MVP buildout" --body "Reproducible setup, closed holes, pytest suite, README + diagrams, demo proof."
```

---

## Self-Review

**Spec coverage:**
- WS1 (setup) → Tasks 1, 2, 3. ✓
- WS2 (baseline run + proof) → Task 10. ✓
- WS3 (close holes) → Tasks 3, 6 (+ side fixes in 8). ✓
- WS4 (extraction + tests) → Tasks 4, 5, 7, 8, 9. ✓
- WS5 (README + diagrams) → Task 11. ✓
- WS6 (verify + push) → Task 12. ✓

**Placeholder scan:** No "TBD/implement later". Where exact existing-code bodies must be moved (geometry, vocabulary, classifier seams), the plan says "move verbatim / read these lines" and pins the new interface with full test code — this is intentional, not a placeholder.

**Type consistency:** `model_paths()` defined in Task 6 is consumed in Task 9. `build_report_row` defined and tested in Task 3. `get_settings()/Settings` defined in Task 2 and consumed in Tasks 3, 6. `frame_interval` flows env → `settings` → `extract_frames`. Names consistent across tasks.

**Known execution-time adaptation:** several seams (`classify`, `get_osha_references`, `_classifier`, exact route handler names) depend on the real bodies in `safety_classifier.py`, `rag_system.py`, and `app.py`; each such task starts with a "read the module" step and instructs aligning names to reality while preserving behavior.
