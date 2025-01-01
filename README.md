# Segment Anything 2 - Core ML

Run Segment Anything 2 (SAM 2) on macOS using Core ML models.

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install coremltools numpy pillow opencv-python
```

## Directory Structure

```
sam2-coreml-python/
├── models/
│   ├── SAM2_1SmallImageEncoderFLOAT16.mlpackage
│   ├── SAM2_1SmallPromptEncoderFLOAT16.mlpackage
│   └── SAM2_1SmallMaskDecoderFLOAT16.mlpackage
├── script.py
└── README.md
```

## Usage

1. Download the Core ML models and place them in the `models` directory
2. Place your input image in the project directory
3. Update the `script.py` file with the input image path
4. Run the script:

```bash
python script.py
```

The script will generate `output_mask.png` containing the segmentation mask.

## Models

The script expects the SAM 2 Core ML models. These need to be downloaded separately and placed in your file system following the above directory structure.

You can find the models here on [Hugging Face](https://huggingface.co/collections/apple/core-ml-segment-anything-2-66e4571a7234dc2560c3db26). I used the `coreml-sam2.1-small` models on my MacBook Pro M3 and the inference time was around 4 seconds. The script will work with other models as well.

## Credit

I took regular inspiration when writing this script from the implementation of [sam2-studio](https://github.com/huggingface/sam2-studio), a SwiftUI app that uses the same Core ML models.
