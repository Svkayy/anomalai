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
