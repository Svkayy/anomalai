#!/usr/bin/env python3
"""
Demo script showing how to use parallel grid segmentation.
"""

import time
from script import SAM2
from app import parallel_grid_segmentation
import cv2
import numpy as np
from PIL import Image

def demo_parallel_segmentation(image_path, model_paths):
    """
    Demonstrate parallel grid segmentation with different configurations.
    """
    print("=" * 60)
    print("SAM2 Parallel Grid Segmentation Demo")
    print("=" * 60)
    
    # Initialize SAM2 model
    sam = SAM2()
    sam.load_models(
        image_encoder_path=model_paths['image_encoder'],
        prompt_encoder_path=model_paths['prompt_encoder'],
        mask_decoder_path=model_paths['mask_decoder']
    )
    
    # Get image embedding once
    sam.get_image_embedding(image_path)
    original_size = Image.open(image_path).size
    
    print(f"Image: {image_path}")
    print(f"Image size: {original_size}")
    print()
    
    # Test different configurations
    configurations = [
        {
            'name': 'Fast (64 points, 2 workers)',
            'target_points': 64,
            'max_workers': 2,
            'batch_size': 16,
            'max_masks': 20
        },
        {
            'name': 'Balanced (256 points, 4 workers)',
            'target_points': 256,
            'max_workers': 4,
            'batch_size': 32,
            'max_masks': 50
        },
        {
            'name': 'High Quality (512 points, 6 workers)',
            'target_points': 512,
            'max_workers': 6,
            'batch_size': 64,
            'max_masks': 100
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing: {config['name']}")
        print("-" * 40)
        
        start_time = time.time()
        
        masks, boxes, areas = parallel_grid_segmentation(
            sam_model=sam,
            filepath=image_path,
            original_size=original_size,
            target_points=config['target_points'],
            max_masks=config['max_masks'],
            max_workers=config['max_workers'],
            batch_size=config['batch_size']
        )
        
        elapsed_time = time.time() - start_time
        
        result = {
            'config': config['name'],
            'time': elapsed_time,
            'masks': len(masks),
            'points': config['target_points'],
            'workers': config['max_workers']
        }
        results.append(result)
        
        print(f"Completed in {elapsed_time:.2f}s")
        print(f"Found {len(masks)} masks from {config['target_points']} points")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time (s)':<10} {'Masks':<8} {'Points':<8} {'Workers':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config']:<30} {result['time']:<10.2f} {result['masks']:<8} "
              f"{result['points']:<8} {result['workers']:<8}")
    
    print()
    print("Performance Tips:")
    print("- Increase max_workers for more CPU cores")
    print("- Increase batch_size for better memory efficiency")
    print("- Increase target_points for more comprehensive coverage")
    print("- Adjust max_masks based on your needs")

if __name__ == "__main__":
    import sys
    import os
    
    # Default paths
    model_paths = {
        'image_encoder': "./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
        'prompt_encoder': "./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
        'mask_decoder': "./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage"
    }
    
    image_path = "./poster.jpg"
    
    # Override with command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)
    
    try:
        demo_parallel_segmentation(image_path, model_paths)
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
