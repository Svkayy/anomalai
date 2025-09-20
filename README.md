# SAM2 CoreML Python - Advanced Image Segmentation Suite

A comprehensive Python implementation of Segment Anything Model 2 (SAM2) using Apple's CoreML framework, featuring a modern web interface, parallel processing, OCR capabilities, and depth estimation.

## 🚀 Features

### Core Segmentation
- **Point-based Segmentation**: Click to add foreground/background points for precise object selection
- **Bounding Box Segmentation**: Draw bounding boxes around objects for automatic segmentation
- **Grid Segmentation**: Automatic full-image segmentation with parallel processing for speed
- **Real-time Preview**: Interactive web interface with live selection preview

### Advanced Processing
- **Parallel Processing**: Multi-threaded grid segmentation for 2-4x speed improvements
- **OCR Integration**: Extract and analyze text from images with font detection
- **Depth Estimation**: Generate depth maps using Depth Anything model
- **Mask Classification**: Automatic object classification using CLIP
- **Export Options**: Download individual masks as PNG files or ZIP archives

### Web Interface
- **Modern UI**: Beautiful, responsive web interface with drag-and-drop upload
- **Interactive Selection**: Click-to-select points and drag-to-draw bounding boxes
- **Real-time Results**: Instant visualization of segmentation results
- **Configuration Panel**: Adjustable parallel processing parameters

## 📋 Prerequisites

- **Python 3.8+**
- **macOS with Apple Silicon** (M1/M2/M3) for optimal CoreML performance
- **Tesseract OCR** (for text extraction features)
- **SAM2 CoreML Models** (see installation instructions below)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sam2-coreml-python
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for text extraction)
```bash
# Using Homebrew (recommended)
brew install tesseract

# Or download from: https://github.com/tesseract-ocr/tesseract
```

### 4. Download SAM2 CoreML Models

#### Option A: Small Model (Faster, Lower Memory)
```bash
# Navigate to the small model directory
cd coreml-sam2.1-small
# Follow the README instructions to download models
```

#### Option B: Large Model (Better Quality, Higher Memory)
```bash
# Navigate to the large model directory  
cd coreml-sam2.1-large
# Follow the README instructions to download models
```

#### Model Directory Structure
After downloading, ensure your `models/` directory contains:
```
models/
├── SAM2_1SmallImageEncoderFLOAT16.mlpackage
├── SAM2_1SmallPromptEncoderFLOAT16.mlpackage
└── SAM2_1SmallMaskDecoderFLOAT16.mlpackage
```

## 🚀 Quick Start

### Web Application (Recommended)
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

### Command Line Interface
```bash
python script.py
```

## 📖 Usage Guide

### Web Application

1. **Upload Image**: Drag and drop or click to upload an image
2. **Choose Method**:
   - **Points**: Click to add exactly 2 points (left=foreground, right=background)
   - **Bounding Box**: Click and drag to draw a rectangle around the object
   - **Grid**: Automatic full-image segmentation with parallel processing
3. **Segment**: Click "🎯 Segment Image" to process
4. **View Results**: See the segmented image with colored overlays

### Command Line Interface

#### Point-based Segmentation
```python
from script import SAM2, PointSelector

# Initialize SAM2
sam = SAM2()
sam.load_models(
    image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
    prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage", 
    mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage"
)

# Select points interactively
point_selector = PointSelector("image.jpg", max_points=2)
points = point_selector.select_points()

# Process segmentation
sam.get_image_embedding("image.jpg")
original_size = Image.open("image.jpg").size
sam.get_prompt_embedding(points, original_size)
mask = sam.get_mask(original_size)
```

#### Bounding Box Segmentation
```python
from script import BoundingBoxSelector

# Select bounding box interactively
bbox_selector = BoundingBoxSelector("image.jpg")
bbox = bbox_selector.select_bounding_box()

# Process segmentation
sam.get_prompt_embedding(bbox, original_size)
mask = sam.get_mask(original_size)
```

#### Parallel Grid Segmentation
```python
from app import parallel_grid_segmentation

# Run parallel grid segmentation
masks, boxes, areas = parallel_grid_segmentation(
    sam_model=sam,
    filepath="image.jpg",
    original_size=(1024, 1024),
    target_points=256,    # Grid density
    max_masks=50,         # Maximum masks to keep
    max_workers=4,        # Parallel workers
    batch_size=32         # Points per batch
)
```

## ⚙️ Configuration Options

### Parallel Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | 4 | Number of parallel workers |
| `batch_size` | 32 | Points processed per batch |
| `target_points` | 256 | Grid points density |
| `max_masks` | 50 | Maximum masks to keep |
| `min_area_frac` | 0.001 | Minimum area as fraction of image |
| `nms_box_thresh` | 0.7 | Non-maximum suppression threshold |
| `dup_mask_iou_thresh` | 0.5 | IoU threshold for deduplication |

### Web API Configuration
```bash
curl -X POST http://localhost:5000/configure_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "max_workers": 6,
    "batch_size": 64,
    "target_points": 512,
    "max_masks": 100
  }'
```

## 🔧 Advanced Features

### OCR and Text Extraction
```bash
# Test OCR functionality
python test_ocr.py

# Extract text from image via web interface
# POST /extract_text/<file_id>
```

Features:
- High-confidence text extraction
- Font analysis and detection
- Character-level mask generation
- Text removal/inpainting
- Bounding box visualization

### Depth Estimation
```bash
# Generate depth map via web interface
# GET /depth/<file_id>
```

Features:
- Depth Anything model integration
- Colored depth visualization
- Real-time depth estimation

### Performance Testing
```bash
# Compare sequential vs parallel processing
python performance_test.py image.jpg \
  ./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage \
  ./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage \
  ./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage

# Demo parallel processing with different configs
python demo_parallel.py image.jpg
```

## 📁 Project Structure

```
sam2-coreml-python/
├── app.py                    # Flask web application
├── script.py                 # Core SAM2 implementation
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web interface template
├── models/                  # SAM2 CoreML models
├── uploads/                 # Temporary uploaded images
├── static/                  # Static web assets
├── coreml-sam2.1-small/     # Small model setup
├── coreml-sam2.1-large/     # Large model setup
├── demo_parallel.py         # Parallel processing demo
├── performance_test.py      # Performance comparison
├── test_ocr.py             # OCR functionality test
├── PARALLEL_PROCESSING.md   # Parallel processing documentation
└── README_WEB.md           # Web app documentation
```

## 🌐 API Endpoints

### Web Application
- `GET /` - Main web interface
- `POST /upload` - Upload image file
- `POST /segment` - Process image segmentation
- `GET /depth/<file_id>` - Generate depth map
- `POST /extract_text/<file_id>` - Extract text from image
- `GET /download_psds/<file_id>` - Download mask ZIP file
- `POST /configure_parallel` - Configure parallel processing

### Request/Response Examples

#### Upload Image
```bash
curl -X POST http://localhost:5000/upload \
  -F "image=@image.jpg"
```

#### Segment with Points
```bash
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "uuid-here",
    "prompt_type": "points",
    "prompts": [
      {"x": 100, "y": 200, "label": 1},
      {"x": 300, "y": 400, "label": 0}
    ]
  }'
```

#### Segment with Bounding Box
```bash
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "uuid-here", 
    "prompt_type": "bbox",
    "prompts": {
      "x1": 100, "y1": 200, "x2": 300, "y2": 400
    }
  }'
```

#### Grid Segmentation
```bash
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "uuid-here",
    "prompt_type": "grid"
  }'
```

## 🚀 Performance Optimization

### Parallel Processing Benefits
- **2-4x speed improvement** with multi-core systems
- **Memory efficient** batch processing
- **Scalable** performance with more CPU cores
- **Configurable** parameters for different use cases

### Recommended Settings
- **max_workers**: Number of CPU cores (or cores - 1)
- **batch_size**: 32-64 depending on available RAM
- **target_points**: 256-512 for good coverage/speed balance
- **max_masks**: 50-100 depending on image complexity

### Memory Management
- Automatic garbage collection between batches
- Memory usage monitoring and reporting
- Efficient numpy array operations
- Configurable batch sizes for memory control

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model paths
ls -la models/

# Verify CoreML compatibility
python -c "import coremltools; print('CoreML available')"
```

#### Tesseract OCR Issues
```bash
# Test Tesseract installation
python test_ocr.py

# Check Tesseract path
which tesseract
```

#### Performance Issues
- **Slow Processing**: Increase `max_workers` or `batch_size`
- **High Memory Usage**: Decrease `batch_size` or `target_points`
- **Poor Mask Quality**: Increase `target_points` or adjust thresholds

#### Web Interface Issues
- **Upload Fails**: Check file size (max 16MB) and format
- **Segmentation Errors**: Verify model loading and image format
- **Browser Issues**: Enable JavaScript and check console for errors

### Debug Mode
```bash
# Run with debug output
python app.py
# Check console for detailed error messages
```

## 📊 Performance Benchmarks

### Typical Performance (M3 MacBook Pro)
- **Point Segmentation**: ~2-3 seconds
- **Bounding Box**: ~2-3 seconds  
- **Grid Segmentation (256 points)**: ~15-20 seconds (parallel)
- **Grid Segmentation (256 points)**: ~45-60 seconds (sequential)
- **Memory Usage**: ~1-2GB peak

### Speed Improvements
- **Parallel vs Sequential**: 2-4x faster
- **Small vs Large Model**: 2-3x faster
- **Optimized vs Default**: 1.5-2x faster

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project uses the SAM2 model which is subject to its own license terms. Please refer to the original SAM2 repository for licensing information.

## 🙏 Acknowledgments

- **Meta AI** for the SAM2 model
- **Apple** for CoreML framework
- **Hugging Face** for model hosting and inspiration
- **OpenCV** and **PIL** for image processing
- **Flask** for the web framework

## 📚 Additional Resources

- [SAM2 Paper](https://arxiv.org/abs/2307.09520)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Parallel Processing Guide](PARALLEL_PROCESSING.md)
- [Web Application Guide](README_WEB.md)

---

**Note**: This implementation is optimized for Apple Silicon Macs. While it may work on Intel Macs, performance will be significantly slower due to CoreML limitations.