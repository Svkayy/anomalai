#!/usr/bin/env python3
"""
Example script showing how to use the labeled segmentation feature.
"""

import requests
import json
import base64
from PIL import Image
import io

def test_labeled_segmentation(image_path, server_url="http://localhost:5000"):
    """
    Test the labeled segmentation feature.
    """
    print("=" * 60)
    print("Testing Labeled Segmentation")
    print("=" * 60)
    
    # Step 1: Upload image
    print("1. Uploading image...")
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{server_url}/upload", files=files)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.json()}")
        return
    
    upload_data = response.json()
    file_id = upload_data['file_id']
    print(f"Image uploaded successfully. File ID: {file_id}")
    
    # Step 2: Run grid segmentation with labels
    print("\n2. Running grid segmentation with classification...")
    segment_data = {
        'file_id': file_id,
        'prompt_type': 'grid'
    }
    
    response = requests.post(f"{server_url}/segment", json=segment_data)
    
    if response.status_code != 200:
        print(f"Segmentation failed: {response.json()}")
        return
    
    result = response.json()
    print(f"Segmentation completed successfully!")
    print(f"Found {result['mask_count']} segments")
    
    # Step 3: Display labels
    print("\n3. Segment Labels:")
    print("-" * 40)
    
    if 'labels' in result and result['labels']:
        for i, label_info in enumerate(result['labels']):
            print(f"Segment {i+1:2d}: {label_info['label']:15s} (confidence: {label_info['confidence']:.3f})")
    else:
        print("No labels available (CLIP model may not be loaded)")
    
    # Step 4: Save segmented image
    print(f"\n4. Saving segmented image...")
    if 'segmented_image' in result:
        # Decode base64 image
        img_data = result['segmented_image'].split(',')[1]  # Remove data:image/png;base64, prefix
        img_bytes = base64.b64decode(img_data)
        
        # Save image
        with open('segmented_with_labels.png', 'wb') as f:
            f.write(img_bytes)
        print("Segmented image saved as 'segmented_with_labels.png'")
    
    return result

def analyze_labels(result):
    """
    Analyze the classification results.
    """
    if 'labels' not in result or not result['labels']:
        print("No labels to analyze")
        return
    
    labels = result['labels']
    
    # Count by label
    label_counts = {}
    for label_info in labels:
        label = label_info['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n" + "=" * 60)
    print("Label Analysis")
    print("=" * 60)
    print(f"Total segments: {len(labels)}")
    print(f"Unique labels: {len(label_counts)}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:15s}: {count:2d} segments")
    
    # High confidence segments
    high_conf = [l for l in labels if l['confidence'] > 0.7]
    print(f"\nHigh confidence segments (>0.7): {len(high_conf)}")
    for label_info in sorted(high_conf, key=lambda x: x['confidence'], reverse=True):
        print(f"  {label_info['label']:15s}: {label_info['confidence']:.3f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_labeling.py <image_path> [server_url]")
        print("Example: python example_labeling.py poster.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000"
    
    try:
        result = test_labeled_segmentation(image_path, server_url)
        if result:
            analyze_labels(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
