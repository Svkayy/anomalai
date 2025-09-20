#!/usr/bin/env python3
"""
Modified app.py with open-vocabulary CLIP integration
"""

# Add this to your existing app.py imports
import openai
import base64
from open_vocab_clip import OpenVocabCLIP, HierarchicalLabelGenerator

# Add these global variables after your existing globals
open_vocab_clip = None
hierarchical_gen = None

def initialize_open_vocab_clip():
    """Initialize open-vocabulary CLIP models"""
    global open_vocab_clip, hierarchical_gen
    try:
        open_vocab_clip = OpenVocabCLIP()
        hierarchical_gen = HierarchicalLabelGenerator()
        print("Open-vocabulary CLIP initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing open-vocabulary CLIP: {e}")
        return False

def classify_mask_open_vocab(image_path, mask, labels=None, use_contextual=True):
    """
    Enhanced classification function with open vocabulary support
    """
    global open_vocab_clip, hierarchical_gen
    
    if open_vocab_clip is None:
        # Fallback to original CLIP if open-vocab not available
        return classify_mask(image_path, mask, labels, None)
    
    try:
        if use_contextual and labels is None:
            # Generate contextual labels based on the image
            labels = open_vocab_clip.generate_contextual_labels(image_path, num_labels=30)
        elif labels is None:
            # Use hierarchical labels for broader coverage
            labels = hierarchical_gen.generate_hierarchical_labels()
        
        # Use open-vocabulary classification
        label, confidence = open_vocab_clip.classify_mask_open_vocab(image_path, mask, labels)
        return label, confidence
        
    except Exception as e:
        print(f"Error in open-vocab classification: {e}")
        # Fallback to original method
        return classify_mask(image_path, mask, labels, None)

def get_dynamic_labels_for_image(image_path, num_labels=30):
    """
    Generate dynamic labels based on image content
    """
    global open_vocab_clip
    
    if open_vocab_clip is None:
        # Return default labels if open-vocab not available
        return [
            "tree", "bush", "grass", "flower", "leaf", "moss",
            "rock", "mountain", "hill", "valley", "cliff", "cave",
            "river", "stream", "lake", "pond", "waterfall", "ocean", "beach", "shore",
            "sky", "cloud", "sun", "moon", "star", "rain", "snow", "fog", "lightning", "rainbow",
            "sand", "soil", "mud", "path", "trail", "road",
            "forest", "meadow", "field", "desert", "swamp", "marsh", "wetland", "tundra",
            "volcano", "glacier", "iceberg",
            "fence", "bridge", "tower", "cabin", "house", "barn", "windmill",
            "animal", "bird", "fish", "insect", "deer", "bear", "wolf", "fox", "rabbit", "horse", "cow", "sheep"
        ]
    
    try:
        return open_vocab_clip.generate_contextual_labels(image_path, num_labels)
    except Exception as e:
        print(f"Error generating dynamic labels: {e}")
        return hierarchical_gen.generate_hierarchical_labels()[:num_labels]

# Modified grid segmentation section for open vocabulary
def process_grid_segmentation_with_open_vocab(sam_model, filepath, original_size, 
                                            target_points=256, max_masks=50, 
                                            min_area_frac=0.001, min_region_area=None,
                                            nms_box_thresh=0.7, dup_mask_iou_thresh=0.5,
                                            max_workers=4, batch_size=32,
                                            use_open_vocab=True, use_contextual_labels=True):
    """
    Enhanced grid segmentation with open-vocabulary classification
    """
    # Get image embedding
    sam_model.get_image_embedding(filepath)
    W, H = original_size

    # Read image for visualization
    original_image = cv2.imread(filepath)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Use parallel processing for grid segmentation
    kept_masks, kept_boxes, kept_areas = parallel_grid_segmentation(
        sam_model=sam_model,
        filepath=filepath,
        original_size=original_size,
        target_points=target_points,
        max_masks=max_masks,
        min_area_frac=min_area_frac,
        nms_box_thresh=nms_box_thresh,
        dup_mask_iou_thresh=dup_mask_iou_thresh,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Classify masks using open-vocabulary CLIP
    mask_labels = []
    if len(kept_masks) > 0:
        print(f"Classifying {len(kept_masks)} masks with open-vocabulary CLIP...")
        
        # Generate dynamic labels for this specific image
        if use_contextual_labels:
            dynamic_labels = get_dynamic_labels_for_image(filepath, num_labels=50)
            print(f"Generated {len(dynamic_labels)} contextual labels for this image")
        else:
            dynamic_labels = None
        
        # Prepare text features once for efficiency
        if use_open_vocab:
            text_features = prepare_text_features(dynamic_labels) if dynamic_labels else None
        else:
            # Use original labels
            LABELS = [
                "tree", "bush", "grass", "flower", "leaf", "moss",
                "rock", "mountain", "hill", "valley", "cliff", "cave",
                "river", "stream", "lake", "pond", "waterfall", "ocean", "beach", "shore",
                "sky", "cloud", "sun", "moon", "star", "rain", "snow", "fog", "lightning", "rainbow",
                "sand", "soil", "mud", "path", "trail", "road",
                "forest", "meadow", "field", "desert", "swamp", "marsh", "wetland", "tundra",
                "volcano", "glacier", "iceberg",
                "fence", "bridge", "tower", "cabin", "house", "barn", "windmill",
                "animal", "bird", "fish", "insect", "deer", "bear", "wolf", "fox", "rabbit", "horse", "cow", "sheep"
            ]
            text_features = prepare_text_features(LABELS)
        
        if text_features is not None:
            for i, mask in enumerate(kept_masks):
                if use_open_vocab:
                    label, confidence = classify_mask_open_vocab(
                        filepath, mask, dynamic_labels, use_contextual_labels
                    )
                else:
                    label, confidence = classify_mask(filepath, mask, dynamic_labels, text_features)
                
                mask_labels.append({
                    "label": label, 
                    "confidence": round(confidence, 3),
                    "mask_index": i
                })
                
                if i % 10 == 0:
                    print(f"Classified {i+1}/{len(kept_masks)} masks")
        else:
            print("Text features not available, skipping classification")
    else:
        print("No masks found for classification")

    return kept_masks, kept_boxes, kept_areas, mask_labels

# Example of how to modify your existing segment_image route
def modified_segment_image_route():
    """
    Example of how to modify your existing /segment route for open vocabulary
    """
    # In your existing segment_image function, replace the grid segmentation part with:
    
    # ... existing code ...
    
    elif prompt_type == 'grid':
        # Enhanced grid segmentation with open vocabulary
        sam_model.get_image_embedding(filepath)
        original_size = Image.open(filepath).size
        W, H = original_size

        # Read image for visualization
        original_image = cv2.imread(filepath)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Use enhanced grid segmentation
        kept_masks, kept_boxes, kept_areas, mask_labels = process_grid_segmentation_with_open_vocab(
            sam_model=sam_model,
            filepath=filepath,
            original_size=original_size,
            target_points=parallel_config['target_points'],
            max_masks=parallel_config['max_masks'],
            min_area_frac=0.001,
            nms_box_thresh=0.7,
            dup_mask_iou_thresh=0.5,
            max_workers=parallel_config['max_workers'],
            batch_size=parallel_config['batch_size'],
            use_open_vocab=True,  # Enable open vocabulary
            use_contextual_labels=True  # Use contextual labels
        )
        
        # ... rest of your existing visualization code ...
        
        return jsonify({
            'success': True,
            'segmented_image': img_base64,
            'mask_count': int(len(kept_masks)),
            'labels': mask_labels,
            'open_vocab_enabled': True
        })

# Add this to your main initialization
def initialize_all_models():
    """Initialize all models including open-vocabulary CLIP"""
    sam_loaded = initialize_sam()
    depth_loaded = initialize_depth()
    clip_loaded = initialize_clip()
    open_vocab_loaded = initialize_open_vocab_clip()
    
    print(f"Model initialization status:")
    print(f"  SAM2: {'✓' if sam_loaded else '✗'}")
    print(f"  Depth: {'✓' if depth_loaded else '✗'}")
    print(f"  CLIP: {'✓' if clip_loaded else '✗'}")
    print(f"  Open-Vocab CLIP: {'✓' if open_vocab_loaded else '✗'}")
    
    return sam_loaded and depth_loaded and (clip_loaded or open_vocab_loaded)

if __name__ == "__main__":
    # Test the open-vocabulary functionality
    print("Testing open-vocabulary CLIP...")
    
    # Initialize models
    success = initialize_all_models()
    
    if success:
        print("All models initialized successfully!")
        
        # Test hierarchical label generation
        if hierarchical_gen:
            labels = hierarchical_gen.generate_hierarchical_labels()
            print(f"Generated {len(labels)} hierarchical labels")
            print("Sample labels:", labels[:10])
    else:
        print("Some models failed to initialize")
