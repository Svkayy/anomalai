"""
Smoke test for the SAM2 model pipeline.

Requires downloaded SAM2 weights — skipped automatically when the weights are
absent. Run explicitly with: pytest -m smoke
"""
import os
import pytest

MODELS = "./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage"


@pytest.mark.smoke
@pytest.mark.skipif(
    not os.path.exists(MODELS),
    reason="SAM2 weights not downloaded",
)
def test_sam_loads():
    from script import SAM2, model_paths

    sam = SAM2()
    # load_models(image_encoder_path, prompt_encoder_path, mask_decoder_path)
    sam.load_models(*model_paths())
    assert sam.image_encoder is not None
    assert sam.prompt_encoder is not None
    assert sam.mask_decoder is not None
