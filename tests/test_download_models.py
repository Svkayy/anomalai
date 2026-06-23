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
