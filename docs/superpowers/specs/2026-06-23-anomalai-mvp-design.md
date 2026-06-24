# anomalai MVP Buildout — Design Spec

**Date:** 2026-06-23
**Status:** Approved (pending written-spec review)
**Repo:** `Svkayy/anomalai`
**Working branch:** `mvp-buildout`

## Context

`anomalai` is a Flask web app that runs a computer-vision workplace-safety
pipeline: an uploaded image/video is segmented with SAM2, segments are labeled
with CLIP, classified safe/dangerous with a BART zero-shot classifier, analyzed
by Gemini for hazards, written up as an OSHA-referenced report via a RAG system,
and persisted to Supabase.

A codebase audit found that ~70% is real, working code, but the project cannot
be run cold and has no tests:

- SAM2 CoreML weights are missing from the repo and from both local copies
  (`models/` and `coreml-sam2.1-small/` are empty everywhere). Source confirmed:
  the public HuggingFace repo [`apple/coreml-sam2.1-small`](https://huggingface.co/apple/coreml-sam2.1-small)
  (3 FLOAT16 `.mlpackage` files: image encoder, prompt encoder, mask decoder).
- No `.env.example`; `setup_database.sql` is likely incomplete.
- `app.py` is a 2,978-line monolith with zero tests.
- Scattered TODOs / dead code / mismatches: `frame_interval` comment says 7 but
  code is 4 (`video_processor.py:48`), macOS-only hardcoded font path
  (`app.py:1225,1433`), commented-out Large-model paths (`script.py:424-426`),
  a runtime "does the `observations` column exist?" fallback
  (`supabase_database.py:99-101`), `print()`-based logging, no bbox validation.

The strongest feature — the portfolio centerpiece — is the **parallel grid
segmentation** (`app.py:634-780`): threaded SAM2 over a 64–256 point grid with
box-NMS + mask-IoU dedup and memory tracking.

## Goal

Bring `anomalai` to a solid, portfolio-credible MVP.

**Definition of done:**
1. The core CV pipeline (SAM2 → CLIP → safety → video) **runs locally on macOS**
   so real demo screenshots / a short screen recording can be captured.
2. Every external integration (Gemini, RAG/Ollama, Supabase) is **genuinely and
   coherently wired** — config-driven via `.env`, error-handled, with no
   TODOs / stubs / dead code / mismatches that a reviewer skimming the repo
   could identify as a hole.
3. A **pytest** suite covers the pure logic and the API/wiring (with all external
   services mocked).
4. A **clean README** with an architecture (pipeline) diagram and a tech-stack
   diagram, plus captured demo proof in `docs/demo/`.

## Guiding principle: close holes, don't camouflage them

We make the real code complete and correct (finish the schema, finish the
integration code, remove dead code and mismatches) so there are no holes to
find — we do **not** fake functionality or fabricate demo output. The cloud code
paths are real and config-driven; they are simply not exercised against live
keys in this pass. All demo screenshots come from code that actually runs locally.

## Explicitly out of scope

- Provisioning live Supabase / Gemini / Ollama infrastructure.
- Production deployment.
- Cross-OS portability beyond macOS (CoreML is the segmentation backend).
- Full rewrite of `app.py` or the front-end UI.
- Exercising cloud integrations against live keys (no keys provided). Their tests
  are structural / mocked only.

## Approach: run-first, then test & polish

Order chosen to de-risk the biggest unknowns (weights, environment) before any
refactoring, and to avoid refactoring working code without a test safety net.

### WS1 — Make it run cold (reproducible setup)
- `scripts/download_models.py`: download `apple/coreml-sam2.1-small` from
  HuggingFace into `models/`, renaming to the exact filenames the code expects
  (`SAM2_1SmallImageEncoderFLOAT16.mlpackage`, `...Prompt...`, `...MaskDecoder...`);
  verify each loads via coremltools.
- `.env.example`: `SUPABASE_URL`, `SUPABASE_KEY`, `GEMINI_API_KEY`, `OLLAMA_HOST`,
  with explanatory comments.
- Complete `setup_database.sql`: full `CREATE TABLE` for `reports`,
  `formal_reports`, `policy_embeddings` (with pgvector) matching
  `supabase_database.py`. This makes the runtime column-existence fallback
  unnecessary, so it is removed.
- Pin Python 3.11; sanity-check `requirements.txt`; add `setup.sh` and a README
  quick-start.

### WS2 — Baseline run + demo proof
- Run the app on a sample workplace image and a short video on macOS.
- Capture annotated outputs, UI screenshots, and a short screen recording into
  `docs/demo/`.
- The Gemini/RAG report panel is rendered via the existing graceful fallback
  (template + OSHA references) so it shows real content without a live key.

### WS3 — Close the holes (coherence pass)
Resolve every audit finding so nothing reads as unfinished:
- Fix `frame_interval` comment/code mismatch; make it configurable.
- Add a cross-platform font fallback instead of the hardcoded macOS path.
- Validate bbox coordinates on input.
- Replace scattered `print()` calls with the `logging` module.
- Convert the commented-out Large-model paths into a clean `MODEL_SIZE`
  (`small`/`large`) config switch.
- Make all cloud integrations uniformly config-driven and error-handled; remove
  the Supabase column-existence runtime fallback (schema now guarantees it).

### WS4 — Targeted extraction + tests
- Extract pure, testable functions out of `app.py` **without behavior change**:
  `geometry.py` (NMS, IoU, coordinate transforms), `vocabulary.py` (workplace
  labels). Keep `safety_classifier.py`, `video_processor.py`, `rag_system.py`,
  `supabase_database.py` as the existing seams.
- pytest suite:
  - **Unit:** NMS / mask-IoU dedup math, coordinate transforms, safety-threshold
    decision logic, vocabulary generation, RAG fallback path, Supabase payload
    construction (mocked client).
  - **API / integration:** Flask test client hitting endpoints with SAM2, CLIP,
    and Gemini mocked — asserts wiring and request/response contracts. No live
    external calls.
  - **Smoke:** a model-gated test (skipped when weights absent) that runs the
    real pipeline on a sample image.
  - `pytest.ini` config; optional GitHub Actions workflow that runs the
    non-model tests.

### WS5 — README + tech-stack diagrams
- Rewrite `README.md`: tagline, demo GIF/screenshots, **architecture (pipeline)
  diagram**, **tech-stack diagram** (generated as editable Excalidraw + SVG),
  setup, usage, testing, project structure.
- Keep the deep-dive docs (`PARALLEL_PROCESSING.md`, `VIDEO_ANALYSIS.md`,
  `LABELING_GUIDE.md`) and link them from the README.

### WS6 — Verify + push
- Tests green, app runs, demo captured → commit and push to `Svkayy/anomalai`.
- Weights are fetched via the download script and not committed (kept out of the
  repo to keep it light).

## Risks & open questions

- **HF filename mismatch:** the files in `apple/coreml-sam2.1-small` may not be
  named exactly as the code expects; the download script handles renaming.
  Verified during WS1.
- **Gemini/RAG screenshots:** rendered via the built-in fallback unless a
  temporary key is dropped in. No fabricated output either way.
- **Refactor risk:** mitigated by extracting only after the baseline is verified
  and alongside the new tests.

## Components & boundaries (target structure)

- `app.py` — Flask routes + orchestration (thinner after extraction).
- `script.py` — SAM2 CoreML wrapper + parallel grid segmentation.
- `geometry.py` *(new)* — NMS, IoU, coordinate transforms (pure functions).
- `vocabulary.py` *(new)* — workplace label vocabulary (pure).
- `safety_classifier.py` — BART zero-shot safe/dangerous classifier.
- `video_processor.py` — frame extraction + per-frame pipeline.
- `rag_system.py` — OSHA RAG report generation (+ fallback).
- `supabase_database.py` — persistence.
- `scripts/download_models.py` *(new)* — model acquisition.
- `tests/` *(new)* — unit + API/integration + smoke.
- `docs/demo/` *(new)* — captured proof.
