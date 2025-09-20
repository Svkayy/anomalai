# Parallel Grid Segmentation for SAM2

This document explains how to use the parallel processing implementation for SAM2 grid segmentation, which significantly speeds up the process by computing masks in parallel.

## Overview

The parallel processing implementation uses Python's `concurrent.futures.ThreadPoolExecutor` to process multiple grid points simultaneously, resulting in substantial speed improvements over the sequential approach.

## Key Features

- **Parallel Processing**: Process multiple grid points simultaneously using thread pools
- **Memory Management**: Batch processing with garbage collection to prevent memory issues
- **Performance Monitoring**: Real-time memory usage and timing information
- **Configurable Parameters**: Adjustable worker count, batch size, and grid density
- **Backward Compatibility**: Drop-in replacement for sequential processing

## Performance Improvements

Typical speedup results:
- **2-4x faster** with 4 workers on multi-core systems
- **Memory efficient** with batch processing and garbage collection
- **Scalable** performance with more CPU cores

## Usage

### Basic Usage

```python
from app import parallel_grid_segmentation
from script import SAM2

# Initialize SAM2 model
sam = SAM2()
sam.load_models(
    image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
    prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
    mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage"
)

# Get image embedding
sam.get_image_embedding("image.jpg")
original_size = (1024, 1024)

# Run parallel grid segmentation
masks, boxes, areas = parallel_grid_segmentation(
    sam_model=sam,
    filepath="image.jpg",
    original_size=original_size,
    target_points=256,      # Number of grid points
    max_masks=50,           # Maximum masks to keep
    max_workers=4,          # Number of parallel workers
    batch_size=32           # Points per batch
)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_points` | 256 | Number of grid points to process |
| `max_masks` | 50 | Maximum number of masks to keep |
| `max_workers` | 4 | Number of parallel workers |
| `batch_size` | 32 | Number of points processed per batch |
| `min_area_frac` | 0.001 | Minimum area as fraction of image |
| `nms_box_thresh` | 0.7 | NMS threshold for boxes |
| `dup_mask_iou_thresh` | 0.5 | IoU threshold for mask deduplication |

### Web API Configuration

You can configure parallel processing parameters via the web API:

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

## Performance Testing

### Run Performance Comparison

```bash
python performance_test.py [image_path] [image_encoder] [prompt_encoder] [mask_decoder]
```

Example:
```bash
python performance_test.py poster.jpg \
  ./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage \
  ./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage \
  ./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage
```

### Run Demo with Different Configurations

```bash
python demo_parallel.py [image_path]
```

Example:
```bash
python demo_parallel.py poster.jpg
```

## Implementation Details

### Parallel Processing Strategy

1. **Grid Generation**: Create a uniform grid of points across the image
2. **Batch Processing**: Process points in batches to manage memory
3. **Parallel Execution**: Use ThreadPoolExecutor for concurrent mask generation
4. **NMS & Deduplication**: Apply non-maximum suppression and IoU-based deduplication
5. **Memory Management**: Force garbage collection between batches

### Memory Optimization

- **Batch Processing**: Process points in small batches to limit memory usage
- **Garbage Collection**: Force garbage collection every few batches
- **Memory Monitoring**: Track memory usage throughout the process
- **Efficient Data Structures**: Use numpy arrays for efficient mask storage

### Thread Safety

The implementation is thread-safe because:
- Each worker processes independent grid points
- SAM2 model inference is thread-safe
- No shared mutable state between workers
- Results are collected and processed sequentially

## Best Practices

### Choosing Parameters

1. **max_workers**: Set to number of CPU cores (or cores - 1)
2. **batch_size**: Start with 32, increase for better memory efficiency
3. **target_points**: More points = better coverage but slower processing
4. **max_masks**: Set based on your specific needs

### Performance Tuning

1. **Monitor Memory Usage**: Watch for memory spikes during processing
2. **Adjust Batch Size**: Increase if you have more RAM, decrease if memory is limited
3. **Worker Count**: More workers = faster processing but more memory usage
4. **Grid Density**: Balance between speed and coverage

### Troubleshooting

**High Memory Usage**:
- Decrease `batch_size`
- Decrease `max_workers`
- Decrease `target_points`

**Slow Processing**:
- Increase `max_workers` (up to CPU core count)
- Increase `batch_size`
- Check if other processes are using CPU

**Poor Mask Quality**:
- Increase `target_points` for better coverage
- Adjust `min_area_frac` to filter out small masks
- Tune NMS and IoU thresholds

## Example Output

```
Processing 256 grid points in parallel with 4 workers...
Initial memory usage: 1024.5 MB
Batch 1/8 completed in 2.34s, kept 12 masks, memory: 1156.2 MB
Batch 2/8 completed in 2.28s, kept 23 masks, memory: 1187.4 MB
...
Parallel grid segmentation completed in 18.45s.
Found 47 valid masks.
Memory usage: 1024.5 MB -> 1203.8 MB (+179.3 MB)
```

## Dependencies

The parallel processing implementation requires:
- `concurrent.futures` (built-in)
- `psutil` (for memory monitoring)
- `numpy` (for efficient array operations)
- `opencv-python` (for image processing)

Install additional dependencies:
```bash
pip install psutil
```

## Future Improvements

Potential enhancements for even better performance:
1. **GPU Acceleration**: Use CUDA for mask processing
2. **Model Optimization**: Quantize models for faster inference
3. **Smart Grid**: Adaptive grid density based on image content
4. **Caching**: Cache image embeddings for multiple operations
5. **Async Processing**: Use asyncio for even better concurrency

## License

This parallel processing implementation follows the same license as the main SAM2 project.
