#!/usr/bin/env python3
"""
Performance test script to compare sequential vs parallel grid segmentation.
"""

import time
import sys
import os
from script import SAM2, Point
from app import parallel_grid_segmentation, mask_to_box, remove_small_regions, box_iou, mask_iou
import numpy as np
import math

def sequential_grid_segmentation(sam_model, filepath, original_size, 
                                target_points=256, max_masks=50, 
                                min_area_frac=0.001, min_region_area=None,
                                nms_box_thresh=0.7, dup_mask_iou_thresh=0.5):
    """
    Sequential version of grid segmentation for comparison.
    """
    W, H = original_size
    
    # Calculate parameters
    points_per_side = int(round(math.sqrt(target_points)))
    points_per_side = max(8, points_per_side)
    
    # Build grid points
    xs = np.linspace(0, W - 1, points_per_side)
    ys = np.linspace(0, H - 1, points_per_side)
    grid_points = [Point(x=float(x), y=float(y), label=1) for y in ys for x in xs]
    
    # Calculate minimum area
    min_area = int(min_area_frac * W * H)
    if min_region_area is None:
        min_region_area = max(16, min_area // 4)
    
    # Shuffle points for better distribution
    rng = np.random.default_rng(123)
    order = np.arange(len(grid_points))
    rng.shuffle(order)
    shuffled_points = [grid_points[i] for i in order]
    
    kept_masks = []
    kept_boxes = []
    kept_areas = []
    
    print(f"Processing {len(grid_points)} grid points sequentially...")
    
    for idx, pt in enumerate(shuffled_points):
        if len(kept_masks) >= max_masks:
            break
            
        # Reset prompt embeddings per point to avoid interference
        sam_model.prompt_embeddings = None
        
        # Prompt and predict
        sam_model.get_prompt_embedding([pt], original_size)
        mask = sam_model.get_mask(original_size)
        
        if mask is None:
            continue
            
        # Ensure binary uint8 {0,1}
        mask = (mask > 0).astype(np.uint8)
        
        # Clean small fragments and holes
        if min_region_area > 0:
            mask = remove_small_regions(mask, min_region_area=min_region_area)
        
        area = int(mask.sum())
        if area < min_area:
            continue
            
        box = mask_to_box(mask)
        if box is None:
            continue
            
        # Box-level NMS check
        too_close = False
        for kb in kept_boxes:
            if box_iou(box, kb) > nms_box_thresh:
                too_close = True
                break
        if too_close:
            continue
            
        # Mask IoU deduplication
        is_dup = False
        for km in kept_masks:
            if mask_iou(mask, km) > dup_mask_iou_thresh:
                is_dup = True
                break
        if is_dup:
            continue
            
        kept_masks.append(mask)
        kept_boxes.append(box)
        kept_areas.append(area)
        
        if idx % 10 == 0:
            print(f"Processed {idx + 1}/{len(shuffled_points)} points, kept {len(kept_masks)} masks")
    
    return kept_masks, kept_boxes, kept_areas

def run_performance_test(image_path, model_paths):
    """
    Run performance comparison between sequential and parallel implementations.
    """
    print("=" * 60)
    print("SAM2 Grid Segmentation Performance Test")
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
    original_size = (1024, 1024)  # Assuming standard size
    
    # Test parameters
    target_points = 64  # Smaller for faster testing
    max_masks = 20
    
    print(f"Image: {image_path}")
    print(f"Target points: {target_points}")
    print(f"Max masks: {max_masks}")
    print()
    
    # Test sequential implementation
    print("Testing Sequential Implementation...")
    start_time = time.time()
    seq_masks, seq_boxes, seq_areas = sequential_grid_segmentation(
        sam, image_path, original_size, 
        target_points=target_points, max_masks=max_masks
    )
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Sequential masks found: {len(seq_masks)}")
    print()
    
    # Test parallel implementation
    print("Testing Parallel Implementation...")
    start_time = time.time()
    par_masks, par_boxes, par_areas = parallel_grid_segmentation(
        sam, image_path, original_size,
        target_points=target_points, max_masks=max_masks,
        max_workers=4, batch_size=16
    )
    par_time = time.time() - start_time
    print(f"Parallel time: {par_time:.2f}s")
    print(f"Parallel masks found: {len(par_masks)}")
    print()
    
    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    print("=" * 60)
    print("RESULTS:")
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel time:   {par_time:.2f}s")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Time saved:     {seq_time - par_time:.2f}s ({((seq_time - par_time) / seq_time * 100):.1f}%)")
    print("=" * 60)
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': speedup,
        'sequential_masks': len(seq_masks),
        'parallel_masks': len(par_masks)
    }

if __name__ == "__main__":
    # Default model paths (adjust as needed)
    model_paths = {
        'image_encoder': "./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
        'prompt_encoder': "./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
        'mask_decoder': "./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage"
    }
    
    # Default image path
    image_path = "./poster.jpg"
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 4:
        model_paths = {
            'image_encoder': sys.argv[2],
            'prompt_encoder': sys.argv[3],
            'mask_decoder': sys.argv[4]
        }
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)
    
    try:
        results = run_performance_test(image_path, model_paths)
    except Exception as e:
        print(f"Error during performance test: {e}")
        sys.exit(1)
