# Segment Labeling with CLIP

This guide explains how to use the automatic segment labeling feature that combines SAM2 segmentation with CLIP classification.

## Overview

The labeling system automatically classifies each segmented region using CLIP (Contrastive Language-Image Pre-training), providing both a label and confidence score for each segment.

## How It Works

1. **Segmentation**: SAM2 creates masks for different regions in the image
2. **Classification**: Each masked region is classified using CLIP against a predefined vocabulary
3. **Labeling**: Returns labels with confidence scores for each segment

## Features

- **Zero-shot Classification**: No training required, works with any image
- **Rich Vocabulary**: 50+ common object and scene labels
- **Confidence Scores**: Each label includes a confidence score (0-1)
- **Parallel Processing**: Classification runs efficiently with the parallel segmentation
- **Fallback Handling**: Gracefully handles cases where CLIP is unavailable

## Supported Labels

The system includes labels for:

### People & Animals
- person, dog, cat, bird, horse, cow, sheep

### Vehicles
- car, truck, bus, motorcycle, bicycle

### Nature
- tree, grass, sky, water

### Buildings & Infrastructure
- building, house, window, door, fence, road, sign, traffic light

### Furniture
- chair, table, bed, sofa, lamp

### Electronics
- phone, laptop, keyboard, mouse, monitor, television

### Food & Kitchen
- bottle, cup, bowl, plate, fork, knife, spoon
- banana, apple, orange, broccoli, carrot, pizza
- donut, cake, sandwich, hot dog, hamburger, french fries

### Objects
- book, book, book, book

## API Usage

### Grid Segmentation with Labels

```bash
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "prompt_type": "grid"
  }'
```

### Response Format

```json
{
  "success": true,
  "segmented_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "mask_count": 26,
  "labels": [
    {
      "label": "person",
      "confidence": 0.892,
      "mask_index": 0
    },
    {
      "label": "car",
      "confidence": 0.756,
      "mask_index": 1
    },
    {
      "label": "tree",
      "confidence": 0.634,
      "mask_index": 2
    }
  ]
}
```

## Python Usage

### Basic Example

```python
import requests
import json

# Upload image
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/upload', files=files)
file_id = response.json()['file_id']

# Run segmentation with labels
segment_data = {
    'file_id': file_id,
    'prompt_type': 'grid'
}
response = requests.post('http://localhost:5000/segment', json=segment_data)
result = response.json()

# Process labels
for i, label_info in enumerate(result['labels']):
    print(f"Segment {i+1}: {label_info['label']} (confidence: {label_info['confidence']:.3f})")
```

### Advanced Analysis

```python
def analyze_segments(result):
    labels = result['labels']
    
    # Group by label
    by_label = {}
    for label_info in labels:
        label = label_info['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(label_info)
    
    # Find high-confidence segments
    high_conf = [l for l in labels if l['confidence'] > 0.8]
    print(f"High confidence segments: {len(high_conf)}")
    
    # Most common labels
    label_counts = {}
    for label_info in labels:
        label = label_info['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    most_common = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most common labels:", most_common[:5])
```

## Configuration

### Custom Labels

You can modify the label vocabulary in `app.py`:

```python
LABELS = [
    "person", "car", "tree", "building", "sky",
    # Add your custom labels here
    "custom_object_1", "custom_object_2"
]
```

### Confidence Threshold

Filter results by confidence:

```python
# Only keep high-confidence labels
high_conf_labels = [
    label for label in result['labels'] 
    if label['confidence'] > 0.7
]
```

## Performance Considerations

### Memory Usage
- CLIP model requires ~1GB RAM
- Classification adds ~2-5 seconds per image
- Consider reducing `max_masks` for faster processing

### Speed Optimization
- Classification runs after segmentation
- Can be disabled by setting `clip_model = None`
- Consider using GPU for faster CLIP inference

### Batch Processing
For multiple images, process them sequentially to avoid memory issues.

## Troubleshooting

### CLIP Not Loading
```
Error loading CLIP model: [error message]
```
- Install CLIP: `pip install clip-by-openai`
- Check PyTorch installation
- Ensure sufficient memory

### Low Confidence Scores
- Try different label vocabulary
- Check image quality and resolution
- Consider preprocessing (contrast, brightness)

### No Labels Returned
- Check if CLIP model loaded successfully
- Verify `clip_model is not None`
- Check server logs for errors

## Example Output

```
Processing 256 grid points in parallel with 4 workers...
Found 26 valid masks.
Classifying 26 masks...
Classified 10/26 masks
Classified 20/26 masks
Classified 26/26 masks

Segment Labels:
Segment  1: person          (confidence: 0.892)
Segment  2: car             (confidence: 0.756)
Segment  3: tree            (confidence: 0.634)
Segment  4: building        (confidence: 0.587)
Segment  5: sky             (confidence: 0.523)
...
```

## Integration with Other Features

### Depth Maps
Combine with depth estimation for 3D understanding:

```python
# Get depth map
depth_response = requests.get(f'http://localhost:5000/depth/{file_id}')
depth_data = depth_response.json()

# Get labeled segments
segment_response = requests.post('http://localhost:5000/segment', json=segment_data)
segment_data = segment_response.json()

# Combine depth and labels
for label_info in segment_data['labels']:
    mask_index = label_info['mask_index']
    # Use mask_index to correlate with depth information
```

### OCR Integration
Combine with text extraction for comprehensive scene understanding:

```python
# Get text
text_response = requests.post(f'http://localhost:5000/extract_text/{file_id}')
text_data = text_response.json()

# Get labeled segments
segment_data = requests.post('http://localhost:5000/segment', json=segment_data).json()

# Combine text and object labels
print("Scene contains:")
print(f"Objects: {[l['label'] for l in segment_data['labels']]}")
print(f"Text: {text_data['text']}")
```

## Best Practices

1. **Use appropriate image resolution** - too small or too large can affect accuracy
2. **Check confidence scores** - filter out low-confidence labels
3. **Combine with other features** - use depth, OCR, and segmentation together
4. **Monitor memory usage** - CLIP can be memory-intensive
5. **Customize vocabulary** - adjust labels for your specific use case

## Future Enhancements

- Custom label training
- Multi-label classification
- Confidence threshold configuration
- Batch processing optimization
- Real-time labeling updates
