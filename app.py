from flask import Flask, render_template, request, jsonify, send_file, make_response
import os
import uuid
import base64
from PIL import Image
import io
import cv2
import numpy as np
from script import SAM2, Point, BoundingBox
import json
import zipfile
import tempfile
import os
from PIL import Image, ImageDraw
import pytesseract
import time
from datetime import datetime
from video_processor import process_video_upload, get_video_frame, get_video_metadata
from safety_classifier import safety_classifier
import requests
from supabase_database import supabase_db_manager, parse_anomalai_observations, parse_anomalai_structured_data
from rag_system import generate_formal_safety_report, is_rag_available
import google.generativeai as genai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for videos

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global progress tracking for video processing
video_progress = {}

# Global error handler for file size exceeded
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': f'File too large. Maximum allowed size is {app.config["MAX_CONTENT_LENGTH"] / (1024*1024):.1f}MB. Please compress your video or use a smaller file.'
    }), 413

# Initialize SAM2 model (global variable)
sam_model = None

# Initialize depth estimation pipeline (global variable)
depth_pipe = None

# Initialize CLIP model for classification (global variables)
clip_model = None
clip_preprocess = None
clip_device = None

# Store video analysis results
video_analysis_results = {}

# Initialize Gemini model
gemini_model = None

# Parallel processing configuration
parallel_config = {
    'max_workers': 4,
    'batch_size': 32,
    'target_points': 32,  # Reduced from 256 to 32 to avoid over-segmentation
    'max_masks': 15       # Reduced from 50 to 15 for cleaner results
}

def initialize_sam():
    """Initialize the SAM2 model"""
    global sam_model
    try:
        sam_model = SAM2()
        sam_model.load_models(
            image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
            prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
            mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
        )
        print("SAM2 model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        return False

def initialize_depth():
    """Initialize the depth estimation pipeline"""
    global depth_pipe
    try:
        depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cpu")
        print("Depth estimation pipeline loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading depth pipeline: {e}")
        return False

def initialize_clip():
    """Initialize the CLIP model for classification"""
    global clip_model, clip_preprocess, clip_device
    try:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
        print(f"CLIP model loaded successfully on {clip_device}")
        return True
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return False

def initialize_gemini():
    """Initialize the Gemini model for safety analysis"""
    global gemini_model
    try:
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Initialize the model
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("Gemini model initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return False


import math
import concurrent.futures
from functools import partial
import psutil
import gc
from transformers import pipeline
import torch
import clip

def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # empty
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def box_area(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1 + 1.0)
    h = max(0.0, y2 - y1 + 1.0)
    return w * h

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0

def mask_iou(a_mask: np.ndarray, b_mask: np.ndarray):
    a = a_mask.astype(bool)
    b = b_mask.astype(bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return (inter / union) if union > 0 else 0.0

def prepare_text_features(labels):
    """Prepare text features for CLIP classification."""
    if clip_model is None:
        return None
        
    try:
        text_tokens = clip.tokenize(labels).to(clip_device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    except Exception as e:
        print(f"Error preparing text features: {e}")
        return None

def classify_mask(image_path, mask, labels, text_features):
    """Classify a SAM mask using CLIP against label vocabulary."""
    if clip_model is None or clip_preprocess is None or text_features is None:
        return "unknown", 0.0
        
    try:
        image = Image.open(image_path).convert("RGB")
        np_img = np.array(image)

        # Apply mask
        masked = np.zeros_like(np_img)
        masked[mask > 0] = np_img[mask > 0]
        pil_patch = Image.fromarray(masked)

        # Preprocess for CLIP
        patch = clip_preprocess(pil_patch).unsqueeze(0).to(clip_device)

        with torch.no_grad():
            image_features = clip_model.encode_image(patch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Similarity with all text labels
            sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            label_idx = sims.argmax().item()
            confidence = sims[0][label_idx].item()

        return labels[label_idx], confidence
    except Exception as e:
        print(f"Error classifying mask: {e}")
        return "unknown", 0.0

def create_segmented_image(image_np, masks):
    """
    Create a segmented image with colored masks
    """
    try:
        # Create a copy of the original image
        segmented_image = image_np.copy()
        
        # Generate colors for each mask
        colors = []
        for i in range(len(masks)):
            # Generate a unique color for each mask
            hue = (i * 137.5) % 360  # Golden angle for good color distribution
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append(color)
        
        # Apply masks with colors
        for i, mask in enumerate(masks):
            color = colors[i]
            # Create colored overlay
            overlay = np.zeros_like(segmented_image)
            overlay[mask > 0] = color
            
            # Blend with original image
            alpha = 0.6  # Transparency
            segmented_image = cv2.addWeighted(segmented_image, 1-alpha, overlay, alpha, 0)
        
        return segmented_image
        
    except Exception as e:
        print(f"Error creating segmented image: {e}")
        return image_np

def process_single_frame_analysis(frame_path, frame_number):
    """
    Process a single frame with segmentation, depth estimation, and classification.
    Returns analysis results for the frame.
    """
    try:
        # Load the frame
        image = Image.open(frame_path)
        W, H = image.size
        
        # Generate depth map
        depth_map = None
        depth_colored = None
        depth_image_base64 = None
        
        if depth_pipe is not None:
            try:
                result = depth_pipe(image)
                depth_map = np.array(result["depth"])
                
                # Generate colored depth map
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                depth_normalized = depth_normalized.astype("uint8")
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                
                # Convert depth map to base64 for display
                depth_pil = Image.fromarray(depth_normalized)
                depth_buffer = io.BytesIO()
                depth_pil.save(depth_buffer, format='PNG')
                depth_image_base64 = base64.b64encode(depth_buffer.getvalue()).decode('utf-8')
                
            except Exception as e:
                print(f"Error generating depth map for frame {frame_number}: {e}")
        
        # Perform segmentation using the same approach as image segmentation
        segmented_image_base64 = None
        objects = []
        
        if sam_model is not None:
            try:
                # Use the same approach as image segmentation
                sam_model.get_image_embedding(frame_path)
                original_size = Image.open(frame_path).size
                W, H = original_size

                # Read image for visualization
                original_image = cv2.imread(frame_path)
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # Use parallel processing for grid segmentation with higher resolution for video analysis
                kept_masks, kept_boxes, kept_areas = parallel_grid_segmentation(
                    sam_model=sam_model,
                    filepath=frame_path,
                    original_size=original_size,
                    target_points=256,  # Higher resolution for video analysis
                    max_masks=30,       # More masks for detailed video analysis
                    min_area_frac=0.001,
                    nms_box_thresh=0.7,
                    dup_mask_iou_thresh=0.5,
                    max_workers=parallel_config['max_workers'],
                    batch_size=parallel_config['batch_size']
                )
                
                if len(kept_masks) > 0:
                    # Generate workplace vocabulary
                    LABELS = generate_workplace_vocabulary()
                    
                    # Prepare text features
                    text_features = prepare_text_features(LABELS)
                    
                    if text_features is not None:
                        # Classify each mask
                        for i, mask in enumerate(kept_masks):
                            label, confidence = classify_mask(frame_path, mask, LABELS, text_features)
                            
                            # Extract coordinates and depth
                            coords = extract_mask_coordinates_and_depth(mask, W, H, depth_map, depth_colored)
                            
                            # Classify for safety
                            safety = safety_classifier.classify_single_object(label)
                            
                            objects.append({
                                "label": label,
                                "confidence": round(confidence, 3),
                                "mask_index": i,  # This is the index in the kept_masks array
                                "coordinates": coords,
                                "safety": safety
                            })
                    
                    # Create segmented image with proper overlays (same as image segmentation)
                    final_image = np.zeros((H, W, 4), dtype=np.uint8)
                    final_image[:, :, :3] = original_rgb
                    final_image[:, :, 3] = 255

                    # Sort by area (largest first) for nicer visualization
                    order = np.argsort(-np.array(kept_areas))
                    colors = [
                        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                        [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
                        [0, 128, 255], [255, 128, 128]
                    ]

                    for i, k in enumerate(order[:30]):  # Up to 30 masks for video analysis
                        mask = kept_masks[k]
                        color = colors[i % len(colors)]
                        overlay = np.zeros((H, W, 4), dtype=np.uint8)
                        overlay[:, :, 0] = color[0]
                        overlay[:, :, 1] = color[1]
                        overlay[:, :, 2] = color[2]
                        overlay[:, :, 3] = 128  # 50% alpha
                        overlay[:, :, 3] = (overlay[:, :, 3] * mask).astype(np.uint8)

                        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                        alpha = np.repeat(alpha, 3, axis=2)
                        final_image[:, :, :3] = (
                            final_image[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha
                        ).astype(np.uint8)

                    # Convert to PIL Image for drawing labels
                    pil_image = Image.fromarray(final_image, 'RGBA')
                    draw = ImageDraw.Draw(pil_image)
                    
                    # Draw labels on each mask
                    for i, k in enumerate(order[:30]):  # Up to 30 masks for video analysis
                        mask = kept_masks[k]
                        color = colors[i % len(colors)]
                        
                        # Find the center of the mask for label placement
                        ys, xs = np.where(mask > 0)
                        if len(xs) > 0 and len(ys) > 0:
                            center_x = int(np.mean(xs))
                            center_y = int(np.mean(ys))
                            
                            # Get label information if available
                            label_text = f"Mask {i+1}"
                            # Find the corresponding label for this mask (k is the original mask index)
                            for obj in objects:
                                if obj.get('mask_index') == k:
                                    label_text = f"{obj['label']} ({obj['confidence']:.2f})"
                                    break
                            
                            # Draw label background
                            bbox = draw.textbbox((center_x, center_y), label_text, font=None)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            
                            # Draw background rectangle
                            padding = 4
                            draw.rectangle([
                                center_x - text_width//2 - padding,
                                center_y - text_height//2 - padding,
                                center_x + text_width//2 + padding,
                                center_y + text_height//2 + padding
                            ], fill=(0, 0, 0, 128))
                            
                            # Draw text
                            draw.text((center_x, center_y), label_text, 
                                    fill=(255, 255, 255, 255), anchor="mm")
                    
                    # Convert to base64
                    segmented_buffer = io.BytesIO()
                    pil_image.save(segmented_buffer, format='PNG')
                    segmented_image_base64 = base64.b64encode(segmented_buffer.getvalue()).decode('utf-8')
                
            except Exception as e:
                print(f"Error processing segmentation for frame {frame_number}: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'frame_number': frame_number,
            'original_image': base64.b64encode(open(frame_path, 'rb').read()).decode('utf-8'),
            'segmented_image': segmented_image_base64,
            'depth_image': depth_image_base64,
            'objects': objects,
            'object_count': len(objects)
        }
        
    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        return None

def generate_workplace_vocabulary():
    """
    Generate simple workplace item vocabulary for object identification.
    Returns a list of basic workplace items for object classification.
    """
    # Simple workplace items for identification
    WORKPLACE_ITEMS = [
        # Furniture & Workspace
        "chair", "desk", "table", "floor", "wall", "ceiling", "door", "window",
        "shelf", "shelving", "cabinet", "drawer", "counter", "workstation",
        
        # Tools & Equipment
        "ladder", "tools", "hammer", "screwdriver", "wrench", "drill", "saw",
        "cart", "trolley", "forklift", "crane", "generator", "compressor",
        
        # Electrical & Utilities
        "electrical wiring", "cables", "extension cord", "outlet", "switch",
        "electrical panel", "conduit", "light", "fan",
        
        # Construction Materials
        "steel", "concrete", "wood", "plastic", "glass", "metal", "pipe",
        "beam", "brick", "tile", "drywall", "insulation",
        
        # Safety & Barriers
        "safety equipment", "helmet", "gloves", "goggles", "vest", "barrier",
        "handrail", "guardrail", "sign", "marking", "tape", "rope",
        
        # Storage & Containers
        "box", "container", "bag", "bucket", "drum", "pallet",
        "cardboard", "packaging", "wrapping",
        
        # Machinery & Vehicles
        "machinery", "equipment", "vehicle", "truck", "tractor", "excavator",
        "bulldozer", "crane", "scaffolding", "platform",
        
        # General Items
        "debris", "clutter", "trash", "waste", "material", "supplies",
        "parts", "components", "hardware", "fixtures"
    ]
    
    return WORKPLACE_ITEMS

def extract_mask_coordinates_and_depth(mask, image_width, image_height, depth_map=None, depth_colored=None):
    """
    Extract relative coordinates (x, y) and depth (z) for a mask.
    
    Args:
        mask: Binary mask array
        image_width: Width of the original image
        image_height: Height of the original image
        depth_map: Optional depth map array (same size as mask)
        depth_colored: Optional colored depth map (BGR format)
    
    Returns:
        dict: Contains relative_x, relative_y, relative_z, center_x, center_y, depth_color
    """
    # Find mask coordinates
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        return {
            "relative_x": 0.0,
            "relative_y": 0.0,
            "relative_z": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "depth_color": "#808080"  # Default gray
        }
    
    # Calculate center coordinates
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    
    # Convert to relative coordinates (0 to 1)
    relative_x = center_x / image_width
    relative_y = center_y / image_height
    
    # Calculate relative depth if depth map is available
    relative_z = 0.0
    depth_color = "#808080"  # Default gray
    
    if depth_map is not None:
        try:
            # Get depth value at the center point of the mask
            center_x_int = int(center_x)
            center_y_int = int(center_y)
            
            # Ensure coordinates are within bounds
            if (0 <= center_x_int < depth_map.shape[1] and 
                0 <= center_y_int < depth_map.shape[0]):
                
                # Get depth value at center point
                center_depth = depth_map[center_y_int, center_x_int]
                
                # Normalize depth to 0-1 range
                min_depth = np.min(depth_map)
                max_depth = np.max(depth_map)
                if max_depth > min_depth:
                    relative_z = (center_depth - min_depth) / (max_depth - min_depth)
                else:
                    relative_z = 0.5  # Default to middle if no depth variation
                    
                # Extract color from colored depth map if available
                if depth_colored is not None:
                    try:
                        if (0 <= center_x_int < depth_colored.shape[1] and 
                            0 <= center_y_int < depth_colored.shape[0]):
                            # Get BGR color at center point
                            bgr_color = depth_colored[center_y_int, center_x_int]
                            # Convert BGR to RGB and then to hex
                            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                            depth_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
                    except Exception as e:
                        print(f"Error extracting depth color: {e}")
                        depth_color = "#808080"
                        
            else:
                print(f"Center coordinates ({center_x_int}, {center_y_int}) out of depth map bounds")
                relative_z = 0.0
        except Exception as e:
            print(f"Error extracting depth at center point: {e}")
            relative_z = 0.0
    
    return {
        "relative_x": round(relative_x, 4),
        "relative_y": round(relative_y, 4),
        "relative_z": round(relative_z, 4),
        "center_x": int(center_x),
        "center_y": int(center_y),
        "depth_color": depth_color
    }


def remove_small_regions(mask: np.ndarray, min_region_area: int = 0):
    """
    Removes very small connected components and fills very small holes.
    Works on boolean/binary mask.
    """
    m = (mask > 0).astype(np.uint8)

    # Remove small foreground components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    cleaned = np.zeros_like(m)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_region_area:
            cleaned[labels == i] = 1

    # Fill small holes by inverting and removing small components
    inv = (1 - cleaned).astype(np.uint8)
    num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = np.zeros_like(inv)
    h_keep = np.ones(num_labels_h, dtype=bool)
    h_keep[0] = True  # background stays
    for i in range(1, num_labels_h):
        if stats_h[i, cv2.CC_STAT_AREA] < min_region_area:
            # this small background region is a hole; fill it
            holes[labels_h == i] = 1
            h_keep[i] = False
    filled = np.clip(cleaned + holes, 0, 1)
    return filled.astype(np.uint8)


def process_single_point(point_data):
    """
    Process a single grid point to generate a mask.
    This function is designed to be called in parallel.
    """
    point, sam_model, original_size, min_area, min_region_area = point_data
    
    try:
        # Reset prompt embeddings to avoid interference
        sam_model.prompt_embeddings = None
        
        # Get prompt embedding for this point
        sam_model.get_prompt_embedding([point], original_size)
        
        # Generate mask
        mask = sam_model.get_mask(original_size)
        
        if mask is None:
            return None
            
        # Ensure binary uint8 {0,1}
        mask = (mask > 0).astype(np.uint8)
        
        # Clean small fragments and holes
        if min_region_area > 0:
            mask = remove_small_regions(mask, min_region_area=min_region_area)
        
        area = int(mask.sum())
        if area < min_area:
            return None
            
        box = mask_to_box(mask)
        if box is None:
            return None
            
        return {
            'mask': mask,
            'box': box,
            'area': area,
            'point': point
        }
        
    except Exception as e:
        print(f"Error processing point {point}: {e}")
        return None


def parallel_grid_segmentation(sam_model, filepath, original_size, 
                              target_points=256, max_masks=50, 
                              min_area_frac=0.001, min_region_area=None,
                              nms_box_thresh=0.7, dup_mask_iou_thresh=0.5,
                              max_workers=4, batch_size=32):
    """
    Perform grid segmentation with parallel processing.
    
    Args:
        sam_model: Initialized SAM2 model
        filepath: Path to the image file
        original_size: (width, height) of the original image
        target_points: Target number of grid points
        max_masks: Maximum number of masks to keep
        min_area_frac: Minimum area as fraction of total image
        min_region_area: Minimum region area for cleaning
        nms_box_thresh: NMS threshold for boxes
        dup_mask_iou_thresh: IoU threshold for mask deduplication
        max_workers: Maximum number of parallel workers
        batch_size: Number of points to process in each batch
    
    Returns:
        tuple: (kept_masks, kept_boxes, kept_areas)
    """
    import time
    start_time = time.time()
    
    W, H = original_size
    
    # Calculate grid points based on resized image dimensions (max 1200x900)
    max_display_width = 1200  # Increased to match frontend original canvas
    max_display_height = 900  # Increased to match frontend original canvas
    
    # Calculate display dimensions maintaining aspect ratio
    display_ratio = min(max_display_width / W, max_display_height / H)
    display_W = int(W * display_ratio)
    display_H = int(H * display_ratio)
    
    # Calculate parameters based on display dimensions
    points_per_side = int(round(math.sqrt(target_points)))
    points_per_side = max(8, points_per_side)
    
    # Build grid points based on display dimensions
    xs = np.linspace(0, display_W - 1, points_per_side)
    ys = np.linspace(0, display_H - 1, points_per_side)
    
    # Scale points back to original image coordinates
    scale_x = W / display_W
    scale_y = H / display_H
    scaled_xs = xs * scale_x
    scaled_ys = ys * scale_y
    
    grid_points = [Point(x=float(x), y=float(y), label=1) for y in scaled_ys for x in scaled_xs]
    
    # Calculate minimum area
    min_area = int(min_area_frac * W * H)
    if min_region_area is None:
        min_region_area = max(16, min_area // 4)
    
    # Shuffle points for better distribution
    rng = np.random.default_rng(123)
    order = np.arange(len(grid_points))
    rng.shuffle(order)
    shuffled_points = [grid_points[i] for i in order]
    
    # Prepare data for parallel processing
    point_data_list = [
        (point, sam_model, original_size, min_area, min_region_area)
        for point in shuffled_points
    ]
    
    # Process points in batches to manage memory
    kept_masks = []
    kept_boxes = []
    kept_areas = []
    
    # Get initial memory usage
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Processing {len(grid_points)} grid points in parallel with {max_workers} workers...")
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Process in batches
    for i in range(0, len(point_data_list), batch_size):
        batch = point_data_list[i:i + batch_size]
        batch_start = time.time()
        
        # Process batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_point, batch))
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        # Apply NMS and deduplication to this batch
        for result in valid_results:
            if len(kept_masks) >= max_masks:
                break
                
            mask = result['mask']
            box = result['box']
            area = result['area']
            
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
        
        # Memory management: force garbage collection every few batches
        if i % (batch_size * 4) == 0:
            gc.collect()
        
        if len(kept_masks) >= max_masks:
            break
            
        batch_time = time.time() - batch_start
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Batch {i//batch_size + 1}/{(len(point_data_list) + batch_size - 1)//batch_size} "
              f"completed in {batch_time:.2f}s, kept {len(kept_masks)} masks, "
              f"memory: {current_memory:.1f} MB")
    
    total_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Parallel grid segmentation completed in {total_time:.2f}s.")
    print(f"Found {len(kept_masks)} valid masks.")
    print(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB "
          f"(+{final_memory - initial_memory:.1f} MB)")
    
    return kept_masks, kept_boxes, kept_areas


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded image
    file.save(filepath)
    
    # Get image dimensions
    with Image.open(filepath) as img:
        width, height = img.size
    
    return jsonify({
        'success': True,
        'file_id': file_id,
        'filename': filename,
        'width': width,
        'height': height
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload and process video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.mov', '.mp4', '.avi', '.mkv', '.webm', '.MOV', '.MP4', '.AVI', '.MKV', '.WEBM'}
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in [ext.lower() for ext in allowed_extensions]:
        return jsonify({'error': f'Unsupported video format: {file_ext}. Supported formats: MOV, MP4, AVI, MKV, WebM'}), 400
    
    try:
        # Generate unique filename
        video_id = str(uuid.uuid4())
        filename = f"{video_id}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded video
        file.save(filepath)
        
        # Check file size after upload
        file_size = os.path.getsize(filepath)
        max_size = app.config['MAX_CONTENT_LENGTH']
        if file_size > max_size:
            os.remove(filepath)  # Clean up
            return jsonify({'error': f'Video file too large: {file_size / (1024*1024):.1f}MB. Maximum allowed: {max_size / (1024*1024):.1f}MB'}), 413
        
        print(f"Processing video: {filename} ({file_size / (1024*1024):.1f}MB)")
        
        # Process video and extract frames
        frames, metadata = process_video_upload(filepath, max_frames=100, video_id=video_id)
        
        if not frames:
            return jsonify({'error': 'Failed to extract frames from video'}), 500
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'filename': filename,
            'frames': frames,
            'metadata': metadata,
            'file_size_mb': round(file_size / (1024*1024), 2)
        })
        
    except Exception as e:
        # Clean up file if it was created
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

@app.route('/video/<video_id>/frame/<int:frame_number>', methods=['GET'])
def get_video_frame_data(video_id, frame_number):
    """Get specific frame data for video analysis"""
    try:
        frame_info = get_video_frame(video_id, frame_number)
        if not frame_info:
            return jsonify({'error': 'Frame not found'}), 404
        
        # Read frame image and convert to base64
        with open(frame_info['path'], 'rb') as f:
            frame_data = f.read()
        
        frame_base64 = base64.b64encode(frame_data).decode('utf-8')
        
        # Determine image format from file extension
        frame_path = frame_info['path']
        if frame_path.lower().endswith('.jpg') or frame_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        
        return jsonify({
            'success': True,
            'frame_info': frame_info,
            'frame_data': f'data:{mime_type};base64,{frame_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get frame: {str(e)}'}), 500

@app.route('/video/<video_id>/metadata', methods=['GET'])
def get_video_metadata_route(video_id):
    """Get video metadata"""
    try:
        metadata = get_video_metadata(video_id)
        if not metadata:
            return jsonify({'error': 'Video metadata not found'}), 404
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get video metadata: {str(e)}'}), 500


@app.route('/segment', methods=['POST'])
def segment_image():
    if sam_model is None:
        return jsonify({'error': 'SAM2 model not loaded'}), 500
    
    data = request.get_json()
    file_id = data.get('file_id')
    prompt_type = data.get('prompt_type')  # 'points' or 'bbox'
    prompts = data.get('prompts')
    
    if not file_id or not prompt_type:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # For grid segmentation, prompts can be empty
    if prompt_type != 'grid' and not prompts:
        return jsonify({'error': 'Missing prompts parameter'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404
    
    try:
        if prompt_type == 'bbox':
            # For bounding box, use numpy array approach similar to SAM2 predictor
            bbox_data = prompts
            x1 = float(bbox_data['x1'])
            y1 = float(bbox_data['y1'])
            x2 = float(bbox_data['x2'])
            y2 = float(bbox_data['y2'])
            
            # Create numpy array box in the format [x1, y1, x2, y2]
            box = np.array([x1, y1, x2, y2])
            
            # Read the original image
            original_image = cv2.imread(filepath)
            original_height, original_width = original_image.shape[:2]
            
            # Convert BGR to RGB for SAM2
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Get image embedding for the original image
            sam_model.get_image_embedding(filepath)
            original_size = Image.open(filepath).size
            
            # Create bounding box object for CoreML compatibility
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            
            # Get prompt embedding for the bounding box
            sam_model.get_prompt_embedding(bbox, original_size)
            
            # Generate mask
            mask = sam_model.get_mask(original_size)
            
            if mask is None:
                return jsonify({'error': 'Failed to generate mask'}), 500
            
            # Create transparent blue overlay
            # Convert BGR to RGB for PIL
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create blue overlay (RGBA)
            blue_overlay = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            blue_overlay[:, :, 0] = 0    # Red
            blue_overlay[:, :, 1] = 0    # Green  
            blue_overlay[:, :, 2] = 255  # Blue
            blue_overlay[:, :, 3] = 128  # Alpha (50% transparency)
            
            # Apply mask to alpha channel
            blue_overlay[:, :, 3] = (blue_overlay[:, :, 3] * mask).astype(np.uint8)
            
            # Convert original image to RGBA
            original_rgba = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            original_rgba[:, :, :3] = original_rgb
            original_rgba[:, :, 3] = 255  # Full opacity
            
            # Blend the images
            alpha = blue_overlay[:, :, 3:4].astype(np.float32) / 255.0
            alpha = np.repeat(alpha, 3, axis=2)
            
            blended = (original_rgba[:, :, :3] * (1 - alpha) + blue_overlay[:, :, :3] * alpha).astype(np.uint8)
            
            # Create final RGBA image
            final_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            final_image[:, :, :3] = blended
            final_image[:, :, 3] = 255  # Full opacity for final image
            
            # Convert to PIL Image for PNG encoding with transparency
            pil_image = Image.fromarray(final_image, 'RGBA')
            
            # Convert to base64 for sending to frontend
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'segmented_image': img_base64,
                'box': box.tolist()  # Include the numpy array box for debugging
            })
        
        elif prompt_type == 'points':
            # For points, perform actual segmentation
            # Get image embedding
            sam_model.get_image_embedding(filepath)
            original_size = Image.open(filepath).size
            
            # Convert points to Point objects (coordinates are already scaled by frontend)
            point_objects = []
            for point_data in prompts:
                point_objects.append(Point(
                    x=float(point_data['x']),
                    y=float(point_data['y']),
                    label=int(point_data['label'])
                ))
            
            sam_model.get_prompt_embedding(point_objects, original_size)
            
            # Generate mask
            mask = sam_model.get_mask(original_size)
            
            if mask is None:
                return jsonify({'error': 'Failed to generate mask'}), 500
            
            # Read the original image
            original_image = cv2.imread(filepath)
            original_height, original_width = original_image.shape[:2]
            
            # Create transparent blue overlay
            # Convert BGR to RGB for PIL
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create blue overlay (RGBA)
            blue_overlay = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            blue_overlay[:, :, 0] = 0    # Red
            blue_overlay[:, :, 1] = 0    # Green  
            blue_overlay[:, :, 2] = 255  # Blue
            blue_overlay[:, :, 3] = 128  # Alpha (50% transparency)
            
            # Apply mask to alpha channel
            blue_overlay[:, :, 3] = (blue_overlay[:, :, 3] * mask).astype(np.uint8)
            
            # Convert original image to RGBA
            original_rgba = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            original_rgba[:, :, :3] = original_rgb
            original_rgba[:, :, 3] = 255  # Full opacity
            
            # Blend the images
            alpha = blue_overlay[:, :, 3:4].astype(np.float32) / 255.0
            alpha = np.repeat(alpha, 3, axis=2)
            
            blended = (original_rgba[:, :, :3] * (1 - alpha) + blue_overlay[:, :, :3] * alpha).astype(np.uint8)
            
            # Create final RGBA image
            final_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            final_image[:, :, :3] = blended
            final_image[:, :, 3] = 255  # Full opacity for final image
            
            # Convert to PIL Image for PNG encoding with transparency
            pil_image = Image.fromarray(final_image, 'RGBA')
            
            # Convert to base64 for sending to frontend
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'segmented_image': img_base64
            })
        
        elif prompt_type == 'grid':
            # --- Parallel SAM-like full-image sweep with filtering & NMS ---
            sam_model.get_image_embedding(filepath)
            original_size = Image.open(filepath).size
            W, H = original_size

            # Read image for visualization
            original_image = cv2.imread(filepath)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Use parallel processing for grid segmentation with global config
            kept_masks, kept_boxes, kept_areas = parallel_grid_segmentation(
                sam_model=sam_model,
                filepath=filepath,
                original_size=original_size,
                target_points=parallel_config['target_points'],
                max_masks=parallel_config['max_masks'],
                min_area_frac=0.001,
                nms_box_thresh=0.7,
                dup_mask_iou_thresh=0.5,
                max_workers=parallel_config['max_workers'],
                batch_size=parallel_config['batch_size']
            )
            
            # Generate depth map for depth extraction
            depth_map = None
            depth_colored = None
            if depth_pipe is not None:
                try:
                    print("Generating depth map for coordinate extraction...")
                    result = depth_pipe(Image.open(filepath))
                    depth_map = np.array(result["depth"])
                    
                    # Generate colored depth map for color extraction
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                    depth_normalized = depth_normalized.astype("uint8")
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                    
                    print("Depth map and colored depth map generated successfully")
                except Exception as e:
                    print(f"Error generating depth map: {e}")
                    depth_map = None
                    depth_colored = None
            else:
                print("Depth estimation model not available, skipping depth extraction")
            
            # Classify masks using CLIP
            mask_labels = []
            if clip_model is not None and len(kept_masks) > 0:
                # Simple workplace items for automatic labeling
                LABELS = generate_workplace_vocabulary()
                print(f"Generated {len(LABELS)} workplace items for classification")

                
                # Prepare text features once
                text_features = prepare_text_features(LABELS)
                
                if text_features is not None:
                    print(f"Classifying {len(kept_masks)} masks...")
                    for i, mask in enumerate(kept_masks):
                        label, confidence = classify_mask(filepath, mask, LABELS, text_features)
                        
                        # Extract coordinates and depth for this mask
                        coords = extract_mask_coordinates_and_depth(mask, W, H, depth_map, depth_colored)
                        
                        mask_labels.append({
                            "label": label, 
                            "confidence": round(confidence, 3),
                            "mask_index": i,
                            "coordinates": coords
                        })
                        if i % 10 == 0:
                            print(f"Classified {i+1}/{len(kept_masks)} masks")
            else:
                print("CLIP model not available, skipping classification")

            # Create visualization
            final_image = np.zeros((H, W, 4), dtype=np.uint8)
            final_image[:, :, :3] = original_rgb
            final_image[:, :, 3] = 255

            # Sort by area (largest first) for nicer visualization
            order = np.argsort(-np.array(kept_areas))
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
                [0, 128, 255], [255, 128, 128]
            ]

            for i, k in enumerate(order[:parallel_config['max_masks']]):
                mask = kept_masks[k]
                color = colors[i % len(colors)]
                overlay = np.zeros((H, W, 4), dtype=np.uint8)
                overlay[:, :, 0] = color[0]
                overlay[:, :, 1] = color[1]
                overlay[:, :, 2] = color[2]
                overlay[:, :, 3] = 128  # 50% alpha
                overlay[:, :, 3] = (overlay[:, :, 3] * mask).astype(np.uint8)

                alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                alpha = np.repeat(alpha, 3, axis=2)
                final_image[:, :, :3] = (
                    final_image[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha
                ).astype(np.uint8)

            # Convert to PIL Image for drawing labels
            pil_image = Image.fromarray(final_image, 'RGBA')
            draw = ImageDraw.Draw(pil_image)
            
            # Draw labels on each mask
            for i, k in enumerate(order[:parallel_config['max_masks']]):
                mask = kept_masks[k]
                color = colors[i % len(colors)]
                
                # Find the center of the mask for label placement
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    center_x = int(np.mean(xs))
                    center_y = int(np.mean(ys))
                    
                    # Get label information if available
                    label_text = f"Mask {i+1}"
                    # Find the corresponding label for this mask (k is the original mask index)
                    for label_info in mask_labels:
                        if label_info['mask_index'] == k:
                            label_text = f"{label_info['label']} ({label_info['confidence']*100:.1f}%)"
                            break
                    
                    # Choose text color (white or black based on background)
                    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                    
                    # Draw text with outline for better visibility
                    try:
                        # Try to use a default font, fallback to basic if not available
                        font_size = max(20, min(H, W) // 25)  # Increased text size
                        try:
                            from PIL import ImageFont
                            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        # Get text bounding box for outline
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Position text at center of mask
                        text_x = center_x - text_width // 2
                        text_y = center_y - text_height // 2
                        
                        # Draw outline (black background)
                        outline_width = 2
                        for dx in range(-outline_width, outline_width + 1):
                            for dy in range(-outline_width, outline_width + 1):
                                if dx*dx + dy*dy <= outline_width*outline_width:
                                    draw.text((text_x + dx, text_y + dy), label_text, 
                                            fill=(0, 0, 0), font=font)
                        
                        # Draw main text
                        draw.text((text_x, text_y), label_text, fill=text_color, font=font)
                        
                    except Exception as e:
                        print(f"Error drawing label for mask {i}: {e}")
                        # Fallback: draw simple text without font
                        draw.text((center_x, center_y), label_text, fill=text_color)

            # Convert the PIL image back to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'segmented_image': img_base64,
                'mask_count': int(len(kept_masks)),
                'labels': mask_labels
            })
        else:
            return jsonify({'error': 'Invalid prompt type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/segment_video_frame', methods=['POST'])
def segment_video_frame():
    """Segment a specific video frame"""
    if sam_model is None:
        return jsonify({'error': 'SAM2 model not loaded'}), 500
    
    data = request.get_json()
    video_id = data.get('video_id')
    frame_number = data.get('frame_number')
    prompt_type = data.get('prompt_type', 'grid')
    prompts = data.get('prompts', [])
    
    if not video_id or frame_number is None:
        return jsonify({'error': 'Missing video_id or frame_number'}), 400
    
    try:
        # Get frame information
        frame_info = get_video_frame(video_id, frame_number)
        if not frame_info:
            return jsonify({'error': 'Frame not found'}), 404
        
        frame_path = frame_info['path']
        
        # Process the frame using existing segmentation logic
        if prompt_type == 'grid':
            # Grid segmentation for video frame
            sam_model.get_image_embedding(frame_path)
            original_size = Image.open(frame_path).size
            W, H = original_size

            # Read image for visualization
            original_image = cv2.imread(frame_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Use parallel processing for grid segmentation
            kept_masks, kept_boxes, kept_areas = parallel_grid_segmentation(
                sam_model=sam_model,
                filepath=frame_path,
                original_size=original_size,
                target_points=parallel_config['target_points'],
                max_masks=parallel_config['max_masks'],
                min_area_frac=0.001,
                nms_box_thresh=0.7,
                dup_mask_iou_thresh=0.5,
                max_workers=parallel_config['max_workers'],
                batch_size=parallel_config['batch_size']
            )
            
            # Generate depth map for depth extraction
            depth_map = None
            depth_colored = None
            if depth_pipe is not None:
                try:
                    print("Generating depth map for video frame coordinate extraction...")
                    result = depth_pipe(Image.open(frame_path))
                    depth_map = np.array(result["depth"])
                    
                    # Generate colored depth map for color extraction
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                    depth_normalized = depth_normalized.astype("uint8")
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                    
                    print("Depth map and colored depth map generated successfully for video frame")
                except Exception as e:
                    print(f"Error generating depth map for video frame: {e}")
                    depth_map = None
                    depth_colored = None
            else:
                print("Depth estimation model not available, skipping depth extraction")
            
            # Classify masks using CLIP
            mask_labels = []
            if clip_model is not None and len(kept_masks) > 0:
                # Simple workplace items for automatic labeling
                LABELS = generate_workplace_vocabulary()
                print(f"Generated {len(LABELS)} workplace items for classification")

                # Prepare text features once
                text_features = prepare_text_features(LABELS)
                
                if text_features is not None:
                    print(f"Classifying {len(kept_masks)} masks for video frame {frame_number}...")
                    for i, mask in enumerate(kept_masks):
                        label, confidence = classify_mask(frame_path, mask, LABELS, text_features)
                        
                        # Extract coordinates and depth for this mask
                        coords = extract_mask_coordinates_and_depth(mask, W, H, depth_map, depth_colored)
                        
                        mask_labels.append({
                            "label": label, 
                            "confidence": round(confidence, 3),
                            "mask_index": i,
                            "coordinates": coords
                        })
                        if i % 10 == 0:
                            print(f"Classified {i+1}/{len(kept_masks)} masks")
                else:
                    print("CLIP model not available, skipping classification")
            else:
                print("CLIP model not available, skipping classification")

            # Create visualization with labels
            final_image = np.zeros((H, W, 4), dtype=np.uint8)
            final_image[:, :, :3] = original_rgb
            final_image[:, :, 3] = 255

            # Sort by area (largest first) for nicer visualization
            order = np.argsort(-np.array(kept_areas))
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
                [0, 128, 255], [255, 128, 128]
            ]

            for i, k in enumerate(order[:parallel_config['max_masks']]):
                mask = kept_masks[k]
                color = colors[i % len(colors)]
                overlay = np.zeros((H, W, 4), dtype=np.uint8)
                overlay[:, :, 0] = color[0]
                overlay[:, :, 1] = color[1]
                overlay[:, :, 2] = color[2]
                overlay[:, :, 3] = 128  # 50% alpha
                overlay[:, :, 3] = (overlay[:, :, 3] * mask).astype(np.uint8)

                alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                alpha = np.repeat(alpha, 3, axis=2)
                final_image[:, :, :3] = (
                    final_image[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha
                ).astype(np.uint8)

            # Convert to PIL Image for drawing labels
            pil_image = Image.fromarray(final_image, 'RGBA')
            draw = ImageDraw.Draw(pil_image)
            
            # Draw labels on each mask
            for i, k in enumerate(order[:parallel_config['max_masks']]):
                mask = kept_masks[k]
                color = colors[i % len(colors)]
                
                # Find the center of the mask for label placement
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    center_x = int(np.mean(xs))
                    center_y = int(np.mean(ys))
                    
                    # Get label information if available
                    label_text = f"Mask {i+1}"
                    # Find the corresponding label for this mask (k is the original mask index)
                    for label_info in mask_labels:
                        if label_info['mask_index'] == k:
                            label_text = f"{label_info['label']} ({label_info['confidence']*100:.1f}%)"
                            break
                    
                    # Choose text color (white or black based on background)
                    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                    
                    # Draw text with outline for better visibility
                    try:
                        # Try to use a default font, fallback to basic if not available
                        font_size = max(20, min(H, W) // 25)  # Increased text size
                        try:
                            from PIL import ImageFont
                            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        # Get text bounding box for outline
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Position text at center of mask
                        text_x = center_x - text_width // 2
                        text_y = center_y - text_height // 2
                        
                        # Draw outline (black background)
                        outline_width = 2
                        for dx in range(-outline_width, outline_width + 1):
                            for dy in range(-outline_width, outline_width + 1):
                                if dx*dx + dy*dy <= outline_width*outline_width:
                                    draw.text((text_x + dx, text_y + dy), label_text, 
                                            fill=(0, 0, 0), font=font)
                        
                        # Draw main text
                        draw.text((text_x, text_y), label_text, fill=text_color, font=font)
                        
                    except Exception as e:
                        print(f"Error drawing label for mask {i}: {e}")
                        # Fallback: draw simple text without font
                        draw.text((center_x, center_y), label_text, fill=text_color)

            # Convert the PIL image back to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'segmented_image': img_base64,
                'mask_count': int(len(kept_masks)),
                'labels': mask_labels,
                'frame_number': frame_number,
                'frame_info': frame_info
            })
        
        else:
            return jsonify({'error': 'Only grid segmentation supported for video frames'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Video frame processing failed: {str(e)}'}), 500

@app.route('/depth_video_frame', methods=['POST'])
def depth_video_frame():
    """Generate depth map for a specific video frame"""
    if depth_pipe is None:
        return jsonify({'error': 'Depth estimation model not loaded'}), 500
    
    data = request.get_json()
    video_id = data.get('video_id')
    frame_number = data.get('frame_number')
    
    if not video_id or frame_number is None:
        return jsonify({'error': 'Missing video_id or frame_number'}), 400
    
    try:
        # Get frame information
        frame_info = get_video_frame(video_id, frame_number)
        if not frame_info:
            return jsonify({'error': 'Frame not found'}), 404
        
        frame_path = frame_info['path']
        
        # Generate depth map using the existing depth estimation
        result = depth_pipe(Image.open(frame_path))
        depth_image = np.array(result["depth"])
        
        # Normalize to [0,255] for visualization
        depth = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
        depth = depth.astype("uint8")

        # Apply inferno colormap (purple/yellow) - same as regular depth endpoint
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # Encode to base64
        _, buffer = cv2.imencode('.png', depth_colored)
        depth_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'depth_image': f'data:image/png;base64,{depth_base64}',
            'frame_number': frame_number
        })
        
    except Exception as e:
        return jsonify({'error': f'Depth estimation failed: {str(e)}'}), 500

@app.route('/depth/<file_id>', methods=['GET'])
def get_depth(file_id):
    """Generate depth map for uploaded image using Depth Anything"""
    if depth_pipe is None:
        return jsonify({'error': 'Depth estimation pipeline not loaded'}), 500
    
    # Handle video frames (stored as .jpg) vs regular images (stored as .png)
    if file_id.startswith('video_'):
        # Parse video frame ID: video_{videoId}_frame_{frameNumber}
        parts = file_id.split('_')
        if len(parts) >= 4 and parts[0] == 'video' and parts[2] == 'frame':
            video_id = parts[1]
            frame_number = int(parts[3])
            frame_info = get_video_frame(video_id, frame_number)
            if not frame_info:
                return jsonify({'error': 'Video frame not found'}), 404
            filepath = frame_info['path']
        else:
            return jsonify({'error': 'Invalid video frame ID format'}), 400
    else:
        # For regular images
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image file not found'}), 404
    
    try:
        # Run depth estimation
        result = depth_pipe(Image.open(filepath))
        depth = np.array(result["depth"])

        # Normalize to [0,255] for visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype("uint8")

        # Apply inferno colormap (purple/yellow)
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # Encode to base64
        _, buffer = cv2.imencode('.png', depth_colored)
        depth_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'depth_image': f'data:image/png;base64,{depth_base64}'
        })

    except Exception as e:
        return jsonify({'error': f'Depth estimation failed: {str(e)}'}), 500


@app.route('/download_psds/<file_id>', methods=['GET'])
def download_psds(file_id):
    """Download PSD files for each mask from grid segmentation"""
    if sam_model is None:
        return jsonify({'error': 'SAM2 model not loaded'}), 500
    
    # Check if the file exists
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404
    
    try:
        # Get image embedding
        sam_model.get_image_embedding(filepath)
        original_size = Image.open(filepath).size
        W, H = original_size
        
        # Read image
        original_image = cv2.imread(filepath)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Use parallel processing for grid segmentation
        kept_masks, kept_boxes, kept_areas = parallel_grid_segmentation(
            sam_model=sam_model,
            filepath=filepath,
            original_size=original_size,
            target_points=parallel_config['target_points'],
            max_masks=10,  # Only keep top 10 most clearly defined masks
            min_area_frac=0.001,
            nms_box_thresh=0.7,
            dup_mask_iou_thresh=0.5,
            max_workers=parallel_config['max_workers'],
            batch_size=parallel_config['batch_size']
        )
        
        if not kept_masks:
            return jsonify({'error': 'No masks found'}), 404
        
        # Create temporary directory for PSD files
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, f"masks_{file_id}.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Sort by area (largest first)
                order = np.argsort(-np.array(kept_areas))
                
                for i, k in enumerate(order):
                    mask = kept_masks[k]
                    area = kept_areas[k]
                    box = kept_boxes[k]
                    
                    # Create masked image with transparency
                    # Convert mask to RGBA
                    mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
                    mask_rgba[:, :, :3] = original_rgb
                    mask_rgba[:, :, 3] = mask * 255  # Alpha channel
                    
                    # Create PIL Image
                    pil_image = Image.fromarray(mask_rgba, 'RGBA')
                    
                    # Save as PNG (PSD equivalent for transparency)
                    filename = f"mask_{i+1:03d}_area_{area}.png"
                    file_path = os.path.join(temp_dir, filename)
                    pil_image.save(file_path, 'PNG')
                    
                    # Add to zip
                    zipf.write(file_path, filename)
            
            # Return the zip file
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f"masks_{file_id}.zip",
                mimetype='application/zip'
            )
    
    except Exception as e:
        return jsonify({'error': f'Failed to create PSD files: {str(e)}'}), 500

@app.route('/extract_text/<file_id>', methods=['POST'])
def extract_text(file_id):
    """Extract text from the uploaded image using Tesseract OCR"""
    if not file_id:
        return jsonify({'error': 'No file ID provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404
    
    try:
        # Read the image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 500
        
        # Convert BGR to RGB for Tesseract
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for better OCR results
        pil_image = Image.fromarray(image_rgb)
        
        # Extract text using Tesseract with detailed data
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Filter words with high confidence (threshold: 60%)
        high_confidence_threshold = 60
        high_confidence_words = []
        high_confidence_confidences = []
        high_confidence_boxes = []
        font_info = {}
        letter_masks = []
        
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            confidence = int(data['conf'][i])
            
            # Only include words with high confidence and non-empty text
            if confidence >= high_confidence_threshold and word and len(word) > 1:
                high_confidence_words.append(word)
                high_confidence_confidences.append(confidence)
                
                # Get bounding box coordinates for this word
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Get font information
                font_name = data['font_name'][i] if 'font_name' in data else 'Unknown'
                font_size = data['font_size'][i] if 'font_size' in data else 'Unknown'
                
                # Store font information
                if font_name not in font_info:
                    font_info[font_name] = {
                        'count': 0,
                        'sizes': set(),
                        'words': []
                    }
                font_info[font_name]['count'] += 1
                font_info[font_name]['sizes'].add(font_size)
                font_info[font_name]['words'].append(word)
                
                # Create letter masks for individual characters
                char_width = w / len(word)
                for char_idx, char in enumerate(word):
                    char_x = x + (char_idx * char_width)
                    char_box = [int(char_x), int(y), int(char_x + char_width), int(y + h)]
                    
                    # Create mask for this character
                    char_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
                    cv2.rectangle(char_mask, (char_box[0], char_box[1]), (char_box[2], char_box[3]), 255, -1)
                    
                    letter_masks.append({
                        'char': char,
                        'box': char_box,
                        'mask': char_mask,
                        'word': word,
                        'confidence': confidence,
                        'font_name': font_name
                    })
                
                high_confidence_boxes.append({
                    'word': word,
                    'confidence': confidence,
                    'box': [x, y, x + w, y + h],
                    'font_name': font_name,
                    'font_size': font_size
                })
        
        # Join high-confidence words
        filtered_text = ' '.join(high_confidence_words)
        
        if not filtered_text:
            return jsonify({
                'success': True,
                'text': 'No high-confidence text found in the image.',
                'confidence': 'N/A',
                'word_count': 0,
                'total_words_found': len([w for w in data['text'] if w.strip()]),
                'high_confidence_threshold': f'{high_confidence_threshold}%',
                'annotated_image': None
            })
        
        # Calculate average confidence for high-confidence words only
        avg_confidence = sum(high_confidence_confidences) / len(high_confidence_confidences) if high_confidence_confidences else 0
        
        # Create annotated image with bounding boxes
        annotated_image = image_rgb.copy()
        
        # Draw bounding boxes around high-confidence words
        for word_info in high_confidence_boxes:
            x1, y1, x2, y2 = word_info['box']
            confidence = word_info['confidence']
            
            # Draw rectangle (green for high confidence)
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add confidence and font label above the box
            font_name = word_info['font_name']
            font_size = word_info['font_size']
            label = f"{word_info['word']} ({confidence}%) - {font_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create letter-removed image (inpaint the text areas)
        letter_removed_image = image_rgb.copy()
        
        # Create a combined mask for all letters
        combined_letter_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        for letter_info in letter_masks:
            combined_letter_mask = cv2.bitwise_or(combined_letter_mask, letter_info['mask'])
        
        # Inpaint the text areas to remove letters
        letter_removed_image = cv2.inpaint(letter_removed_image, combined_letter_mask, 3, cv2.INPAINT_TELEA)
        
        # Convert letter-removed image to base64
        _, buffer_removed = cv2.imencode('.png', cv2.cvtColor(letter_removed_image, cv2.COLOR_RGB2BGR))
        letter_removed_image_base64 = base64.b64encode(buffer_removed).decode('utf-8')
        
        # Create letter mask visualization (white letters on black background)
        letter_mask_visualization = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
        for letter_info in letter_masks:
            x1, y1, x2, y2 = letter_info['box']
            cv2.rectangle(letter_mask_visualization, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # Convert letter mask visualization to base64
        _, buffer_mask = cv2.imencode('.png', letter_mask_visualization)
        letter_mask_base64 = base64.b64encode(buffer_mask).decode('utf-8')
        
        # Process font information for output
        font_summary = []
        for font_name, info in font_info.items():
            sizes = sorted(list(info['sizes'])) if info['sizes'] != {'Unknown'} else ['Unknown']
            font_summary.append({
                'name': font_name,
                'count': info['count'],
                'sizes': sizes,
                'sample_words': info['words'][:3]  # Show first 3 words as examples
            })
        
        return jsonify({
            'success': True,
            'text': filtered_text,
            'confidence': f'{avg_confidence:.1f}%',
            'word_count': len(high_confidence_words),
            'total_words_found': len([w for w in data['text'] if w.strip()]),
            'high_confidence_threshold': f'{high_confidence_threshold}%',
            'annotated_image': f'data:image/png;base64,{annotated_image_base64}',
            'font_analysis': font_summary,
            'letter_removed_image': f'data:image/png;base64,{letter_removed_image_base64}',
            'letter_mask_visualization': f'data:image/png;base64,{letter_mask_base64}',
            'letter_count': len(letter_masks)
        })
        
    except Exception as e:
        return jsonify({'error': f'Text extraction failed: {str(e)}'}), 500

@app.route('/configure_parallel', methods=['POST'])
def configure_parallel():
    """Configure parallel processing parameters"""
    data = request.get_json()
    
    # Update global configuration (you might want to store this in a config file or database)
    global parallel_config
    parallel_config = {
        'max_workers': data.get('max_workers', 4),
        'batch_size': data.get('batch_size', 32),
        'target_points': data.get('target_points', 32),
        'max_masks': data.get('max_masks', 15)
    }
    
    return jsonify({
        'success': True,
        'config': parallel_config
    })

@app.route('/api/classify_safety', methods=['POST'])
def classify_safety():
    """Classify objects as safe or unsafe using zero-shot classification"""
    try:
        data = request.get_json()
        objects_data = data.get('objects', [])
        alpha = data.get('alpha', 0.05)
        
        if not objects_data:
            return jsonify({'error': 'No objects provided'}), 400
        
        # Classify objects for safety
        classified_objects = safety_classifier.classify_objects(objects_data, alpha)
        
        return jsonify({
            'classified_objects': classified_objects,
            'total_objects': len(classified_objects),
            'dangerous_count': sum(1 for obj in classified_objects if obj.get('safety', {}).get('is_dangerous', False)),
            'safe_count': sum(1 for obj in classified_objects if not obj.get('safety', {}).get('is_dangerous', False))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_video_analysis_json', methods=['POST'])
def download_video_analysis_json():
    """Download comprehensive JSON data for video analysis results"""
    try:
        global video_analysis_results
        
        data = request.get_json()
        video_id = data.get('video_id')
        
        if not video_id:
            return jsonify({'error': 'Missing video_id'}), 400
        
        # Get video metadata
        video_metadata = get_video_metadata(video_id)
        if not video_metadata:
            return jsonify({'error': 'Video not found'}), 404
        
        # Get video analysis results
        analysis_results = video_analysis_results.get(video_id)
        if not analysis_results:
            return jsonify({'error': 'No analysis results found for this video'}), 404
        
        # Build comprehensive JSON structure
        json_data = {
            "video_metadata": {
                "video_id": video_id,
                "device_type": "smart glasses",  # Default as requested
                "total_frames": video_metadata.get('total_frames', 0),
                "extracted_frames": video_metadata.get('extracted_frames', 0),
                "fps": video_metadata.get('fps', 30),
                "duration_seconds": video_metadata.get('duration_seconds', 0),
                "width": video_metadata.get('width', 0),
                "height": video_metadata.get('height', 0),
                "analysis_timestamp": analysis_results.get('analysis_timestamp'),
                "frames_processed": analysis_results.get('frames_processed', 0),
                "total_processing_time": analysis_results.get('total_processing_time', 0),
                "note": "Only frames with unsafe objects are included in this export. No image data is included."
            },
            "frames": []
        }
        
        # Process each frame - only include frames with unsafe objects
        unsafe_frames_count = 0
        for frame_data in analysis_results.get('frames', []):
            # Filter objects to only include unsafe ones
            unsafe_objects = []
            for obj in frame_data.get('objects', []):
                safety = obj.get('safety', {})
                is_safe = safety.get('is_safe', True)
                
                # Only include unsafe objects
                if not is_safe:
                    coords = obj.get('coordinates', {})
                    
                    object_info = {
                        "label": obj.get('label', 'unknown'),
                        "confidence": obj.get('confidence', 0),
                        "coordinates": {
                            "x": coords.get('relative_x', 0),
                            "y": coords.get('relative_y', 0),
                            "z": coords.get('relative_z', 0),
                            "depth_color": coords.get('depth_color', '#808080')
                        },
                        "safety": {
                            "classification": safety.get('classification', 'unknown'),
                            "confidence": safety.get('confidence', 0),
                            "is_safe": safety.get('is_safe', False)
                        },
                        "mask_index": obj.get('mask_index', 0)
                    }
                    
                    unsafe_objects.append(object_info)
            
            # Only include frames that have at least one unsafe object
            if unsafe_objects:
                frame_info = {
                    "frame_number": frame_data.get('frame_number'),
                    "processing_time": frame_data.get('processing_time', 0),
                    "objects": unsafe_objects
                }
                
                json_data["frames"].append(frame_info)
                unsafe_frames_count += 1
        
        # Update metadata with unsafe frames count
        json_data["video_metadata"]["unsafe_frames_count"] = unsafe_frames_count
        
        # Create JSON string
        json_string = json.dumps(json_data, indent=2)
        
        # Create response with JSON file
        response = make_response(json_string)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=video_analysis_{video_id}.json'
        
        return response
        
    except Exception as e:
        print(f"Error generating video analysis JSON: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate JSON download'}), 500

@app.route('/api/analyze_video_with_gemini', methods=['POST'])
def analyze_video_with_gemini():
    """Analyze video frames with unsafe objects using Gemini"""
    try:
        global video_analysis_results, gemini_model
        
        data = request.get_json()
        video_id = data.get('video_id')
        
        if not video_id:
            return jsonify({'error': 'Missing video_id'}), 400
        
        if gemini_model is None:
            return jsonify({'error': 'Gemini model not initialized'}), 500
        
        # Get video analysis results
        analysis_results = video_analysis_results.get(video_id)
        if not analysis_results:
            return jsonify({'error': 'No analysis results found for this video'}), 404
        
        # Get video metadata
        video_metadata = get_video_metadata(video_id)
        if not video_metadata:
            return jsonify({'error': 'Video not found'}), 404
        
        # Build reduced JSON structure (no video metadata)
        reduced_json_data = {
            "frames": []
        }
        
        # Process each frame - only include frames with unsafe objects
        unsafe_frames_count = 0
        gemini_analyses = []
        
        for frame_data in analysis_results.get('frames', []):
            # Filter objects to only include unsafe ones
            unsafe_objects = []
            for obj in frame_data.get('objects', []):
                safety = obj.get('safety', {})
                is_safe = safety.get('is_safe', True)
                
                # Only include unsafe objects
                if not is_safe:
                    coords = obj.get('coordinates', {})
                    
                    object_info = {
                        "label": obj.get('label', 'unknown'),
                        "confidence": obj.get('confidence', 0),
                        "coordinates": {
                            "x": coords.get('relative_x', 0),
                            "y": coords.get('relative_y', 0),
                            "z": coords.get('relative_z', 0),
                            "depth_color": coords.get('depth_color', '#808080')
                        },
                        "safety": {
                            "classification": safety.get('classification', 'unknown'),
                            "confidence": safety.get('confidence', 0),
                            "is_safe": safety.get('is_safe', False)
                        },
                        "mask_index": obj.get('mask_index', 0)
                    }
                    
                    unsafe_objects.append(object_info)
            
            # Only process frames that have at least one unsafe object
            if unsafe_objects:
                frame_number = frame_data.get('frame_number')
                original_image = frame_data.get('original_image')
                
                # Add frame info to reduced JSON
                frame_info = {
                    "frame_number": frame_number,
                    "processing_time": frame_data.get('processing_time', 0),
                    "objects": unsafe_objects
                }
                reduced_json_data["frames"].append(frame_info)
                unsafe_frames_count += 1
                
                # Generate AnomalAI analysis for this frame
                try:
                    # Convert base64 to PIL Image
                    import base64
                    from io import BytesIO
                    
                    image_data = base64.b64decode(original_image)
                    image = Image.open(BytesIO(image_data))
                    
                    # First, get initial hazard detection
                    initial_prompt = """Analyze this image for workplace safety hazards. Output rules: Return up to 4–5 bullet points maximum. Each bullet point must describe a distinct hazard, stated factually and neutrally. Identify both obvious risks (e.g., missing PPE, vehicle proximity, exposed machinery) and subtle risks (e.g., dangling wires, misplaced objects, obstructed walkways, poor visibility). Do not include headers, labels, or commentary. If no hazards are visible, return nothing. Do not invent hazards not supported by the image."""
                    
                    initial_response = gemini_model.generate_content([initial_prompt, image])
                    initial_analysis = initial_response.text
                    
                    if initial_analysis and initial_analysis.strip():
                        # Generate AnomalAI structured analysis
                        anomalai_prompt = f"""You are AnomalAI, an AI safety assistant. 
Your purpose is to analyze workplace images or descriptions and generate concise, neutral safety observations. 
Always be factual and only describe hazards that are clearly mentioned in the input. 
Each distinct hazard must map to its own observation with fields for label, severity, reasons, actions, tags, actors, timeframe, and frames. 
Also generate a brief natural description of the unsafe scene and a summary report with counts of low, medium, and high hazards. 
Do not include irrelevant commentary, do not fabricate hazards, and do not reference policies or external documents.

You are AnomalAI, an assistant that turns workplace safety image descriptions into structured safety observations.

INPUT: <<<{initial_analysis}>>>

TASK:
1. Read the text carefully and extract each distinct hazard as a separate observation.
2. For each observation, output:
   - label: a short snake_case identifier (e.g., improper_ladder_storage, cluttered_walkway, loose_wires).
   - severity: choose one of "low", "medium", or "high" based on the hazard's risk.
   - reasons: 1–3 short bullet phrases explaining why it is unsafe.
   - actions: 1–3 short recommended corrective actions.
   - tags: 2–5 short keywords.
   - actors: list of entities involved (e.g., ["person#1", "forklift#7"]). If none mentioned, return an empty list.
   - timeframe: set to "image" (since this came from a still photo).
   - frames: set to "f{frame_number:04d}".

OUTPUT:
- First, give a concise 2–3 line natural description of what is unsafe overall.
- Then output the JSON for the report object and an array of observations as described above.

RULES:
- Be factual and neutral; do not invent hazards that are not present in the input.
- Do not add extra commentary beyond the description + JSON."""

                        anomalai_response = gemini_model.generate_content(anomalai_prompt)
                        structured_analysis = anomalai_response.text
                        
                        gemini_analyses.append({
                            "frame_number": frame_number,
                            "initial_analysis": initial_analysis,
                            "structured_analysis": structured_analysis,
                            "objects": unsafe_objects
                        })
                    else:
                        # No hazards detected in this frame
                        gemini_analyses.append({
                            "frame_number": frame_number,
                            "initial_analysis": "No hazards detected",
                            "structured_analysis": None,
                            "objects": unsafe_objects
                        })
                        
                except Exception as e:
                    print(f"Error analyzing frame {frame_number} with Gemini: {e}")
                    gemini_analyses.append({
                        "frame_number": frame_number,
                        "initial_analysis": f"Error: {str(e)}",
                        "structured_analysis": None,
                        "objects": unsafe_objects
                    })
        
        # Build final response with video metadata
        final_response = {
            "video_id": video_id,
            "video_metadata": {
                "video_id": video_id,
                "total_frames": video_metadata.get('total_frames', 0),
                "extracted_frames": video_metadata.get('extracted_frames', 0),
                "frame_interval": video_metadata.get('frame_interval', 7),
                "duration_seconds": video_metadata.get('duration_seconds', 0),
                "device_type": "smart glasses",
                "analysis_timestamp": time.time()
            },
            "unsafe_frames_count": unsafe_frames_count,
            "gemini_analyses": gemini_analyses,
            "reduced_json": reduced_json_data
        }
        
        # Update report in database with observation counts
        if supabase_db_manager.is_available():
            try:
                # Get the report_id from the analysis results
                report_id = analysis_results.get('report_id')
                
                if report_id:
                    # Parse all AnomalAI analyses to get total observation counts and structured data
                    total_observations = {
                        'total_observations': 0,
                        'low': 0,
                        'medium': 0,
                        'high': 0
                    }
                    
                    # Collect all structured observations data
                    all_structured_observations = {
                        'video_analysis_summary': {
                            'total_frames_analyzed': len(gemini_analyses),
                            'unsafe_frames_count': len([a for a in gemini_analyses if a.get('structured_analysis')]),
                            'analysis_timestamp': datetime.now().isoformat()
                        },
                        'frame_analyses': []
                    }
                    
                    for analysis in gemini_analyses:
                        if analysis.get('structured_analysis'):
                            # Parse observation counts
                            observations = parse_anomalai_observations(analysis['structured_analysis'])
                            total_observations['total_observations'] += observations['total_observations']
                            total_observations['low'] += observations['low']
                            total_observations['medium'] += observations['medium']
                            total_observations['high'] += observations['high']
                            
                            # Parse structured data
                            structured_data = parse_anomalai_structured_data(analysis['structured_analysis'])
                            frame_analysis = {
                                'frame_number': analysis.get('frame_number', 0),
                                'frame_timestamp': analysis.get('frame_timestamp', 0),
                                'description': structured_data.get('description', ''),
                                'summary': structured_data.get('summary', {'low': 0, 'medium': 0, 'high': 0}),
                                'observations': structured_data.get('observations', [])
                            }
                            all_structured_observations['frame_analyses'].append(frame_analysis)
                    
                    # Update the report in database with both counts and structured data
                    success = supabase_db_manager.update_report_observations(
                        report_id, 
                        total_observations, 
                        all_structured_observations
                    )
                    if success:
                        print(f"Report updated with observations: {total_observations}")
                        final_response['report_updated'] = True
                        final_response['observation_counts'] = total_observations
                        
                        # Automatically generate formal report after video analysis
                        try:
                            print(f"Generating formal report for video analysis: {report_id}")
                            formal_report_content = generate_formal_safety_report(all_structured_observations)
                            
                            if formal_report_content and "Error generating" not in formal_report_content:
                                # Get the original report data for formal report creation
                                original_report = supabase_db_manager.get_report(report_id)
                                if original_report:
                                    formal_report_id = supabase_db_manager.create_formal_report(
                                        report_id=report_id,
                                        video_duration=original_report['video_duration'],
                                        video_captured_at=datetime.fromisoformat(original_report['video_captured_at'].replace('Z', '+00:00')),
                                        video_device_type=original_report['video_device_type'],
                                        total_observations=total_observations['total_observations'],
                                        low=total_observations['low'],
                                        medium=total_observations['medium'],
                                        high=total_observations['high'],
                                        formal_report_content=formal_report_content
                                    )
                                    print(f"Formal report created successfully: {formal_report_id}")
                                    final_response['formal_report_id'] = formal_report_id
                                    final_response['formal_report_generated'] = True
                                else:
                                    print("Could not retrieve original report for formal report creation")
                                    final_response['formal_report_generated'] = False
                            else:
                                print(f"Failed to generate formal report: {formal_report_content}")
                                final_response['formal_report_generated'] = False
                        except Exception as e:
                            print(f"Error generating formal report: {e}")
                            final_response['formal_report_generated'] = False
                    else:
                        print(f"Failed to update report {report_id}")
                        final_response['report_updated'] = False
                else:
                    print("No report_id found in analysis results")
                    final_response['report_updated'] = False
                    
            except Exception as e:
                print(f"Error updating report in Supabase: {e}")
                final_response['report_updated'] = False
                final_response['db_error'] = str(e)
        else:
            print("Supabase not available - skipping report update")
            final_response['report_updated'] = False
        
        return jsonify(final_response)
        
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to analyze with Gemini'}), 500

@app.route('/api/analyze_image_with_gemini', methods=['POST'])
def analyze_image_with_gemini():
    """Analyze a single image with Gemini for safety hazards"""
    try:
        global gemini_model
        
        # Initialize Gemini if not already done
        if gemini_model is None:
            if not initialize_gemini():
                return jsonify({'error': 'Failed to initialize Gemini model'}), 500
        
        # Get the image data from the request
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Convert base64 to PIL Image
        import base64
        from io import BytesIO
        
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Create prompt for Gemini
        prompt = """Analyze this image for workplace safety hazards. Output rules: Return up to 4–5 bullet points maximum. Each bullet point must describe a distinct hazard, stated factually and neutrally. Identify both obvious risks (e.g., missing PPE, vehicle proximity, exposed machinery) and subtle risks (e.g., dangling wires, misplaced objects, obstructed walkways, poor visibility). Do not include headers, labels, or commentary. If no hazards are visible, return nothing. Do not invent hazards not supported by the image."""
        
        # Send to Gemini for initial hazard detection
        try:
            response = gemini_model.generate_content([prompt, image])
            initial_analysis = response.text
        except Exception as e:
            return jsonify({'error': f'Gemini analysis failed: {str(e)}'}), 500
        
        # If no hazards detected, return simple response
        if not initial_analysis or initial_analysis.strip() == '':
            return jsonify({
                'success': True,
                'gemini_analysis': '✅ No safety hazards detected in this image.',
                'structured_analysis': None,
                'timestamp': time.time()
            })
        
        # If hazards detected, send to AnomalAI for structured analysis
        try:
            anomalai_prompt = f"""You are AnomalAI, an AI safety assistant. 
Your purpose is to analyze workplace images or descriptions and generate concise, neutral safety observations. 
Always be factual and only describe hazards that are clearly mentioned in the input. 
Each distinct hazard must map to its own observation with fields for label, severity, reasons, actions, tags, actors, timeframe, and frames. 
Also generate a brief natural description of the unsafe scene and a summary report with counts of low, medium, and high hazards. 
Do not include irrelevant commentary, do not fabricate hazards, and do not reference policies or external documents.

You are AnomalAI, an assistant that turns workplace safety image descriptions into structured safety observations.

INPUT: <<<{initial_analysis}>>>

TASK:
1. Read the text carefully and extract each distinct hazard as a separate observation.
2. For each observation, output:
   - label: a short snake_case identifier (e.g., improper_ladder_storage, cluttered_walkway, loose_wires).
   - severity: choose one of "low", "medium", or "high" based on the hazard's risk.
   - reasons: 1–3 short bullet phrases explaining why it is unsafe.
   - actions: 1–3 short recommended corrective actions.
   - tags: 2–5 short keywords.
   - actors: list of entities involved (e.g., ["person#1", "forklift#7"]). If none mentioned, return an empty list.
   - timeframe: set to "image" (since this came from a still photo).
   - frames: set to "f0001".

OUTPUT:
- First, give a concise 2–3 line natural description of what is unsafe overall.
- Then output the JSON for the report object and an array of observations as described above.

RULES:
- Be factual and neutral; do not invent hazards that are not present in the input.
- Do not add extra commentary beyond the description + JSON."""

            anomalai_response = gemini_model.generate_content(anomalai_prompt)
            structured_analysis = anomalai_response.text
            
        except Exception as e:
            print(f"AnomalAI analysis failed: {e}")
            # Fall back to original analysis if AnomalAI fails
            structured_analysis = None
        
        return jsonify({
            'success': True,
            'gemini_analysis': initial_analysis,
            'structured_analysis': structured_analysis,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Error in image Gemini analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to analyze image with Gemini'}), 500

@app.route('/api/generate_formal_report', methods=['POST'])
def generate_formal_report():
    """Generate a formal safety report using RAG system from existing observations"""
    try:
        data = request.get_json()
        report_id = data.get('report_id')
        
        if not report_id:
            return jsonify({'error': 'Missing report_id'}), 400
        
        # Check if RAG system is available
        if not is_rag_available():
            return jsonify({'error': 'RAG system not available. Please ensure all dependencies are installed.'}), 500
        
        # Get the original report from database
        if not supabase_db_manager.is_available():
            return jsonify({'error': 'Database not available'}), 500
        
        original_report = supabase_db_manager.get_report(report_id)
        if not original_report:
            return jsonify({'error': 'Report not found'}), 404
        
        # Extract observations data
        observations_data = None
        if 'observations' in original_report and original_report['observations']:
            observations_data = original_report['observations']
        elif 'description' in original_report and original_report['description']:
            # Try to parse from description column (workaround)
            try:
                import json
                observations_data = json.loads(original_report['description'])
            except json.JSONDecodeError:
                return jsonify({'error': 'No valid observations data found in report'}), 400
        else:
            return jsonify({'error': 'No observations data found in report'}), 400
        
        # Generate formal safety report using RAG
        print(f"Generating formal report for report_id: {report_id}")
        formal_report_content = generate_formal_safety_report(observations_data)
        
        if not formal_report_content or "Error generating" in formal_report_content:
            return jsonify({'error': f'Failed to generate formal report: {formal_report_content}'}), 500
        
        # Create formal report in database
        try:
            formal_report_id = supabase_db_manager.create_formal_report(
                report_id=report_id,
                video_duration=original_report['video_duration'],
                video_captured_at=datetime.fromisoformat(original_report['video_captured_at'].replace('Z', '+00:00')),
                video_device_type=original_report['video_device_type'],
                total_observations=original_report['total_observations'],
                low=original_report['low'],
                medium=original_report['medium'],
                high=original_report['high'],
                formal_report_content=formal_report_content
            )
            
            return jsonify({
                'success': True,
                'formal_report_id': formal_report_id,
                'formal_report_content': formal_report_content,
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error creating formal report in database: {e}")
            return jsonify({
                'success': True,
                'formal_report_id': None,
                'formal_report_content': formal_report_content,
                'generated_at': datetime.now().isoformat(),
                'warning': 'Formal report generated but not saved to database'
            })
        
    except Exception as e:
        print(f"Error in formal report generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate formal report'}), 500

@app.route('/api/get_formal_report/<formal_report_id>', methods=['GET'])
def get_formal_report(formal_report_id):
    """Retrieve a formal report from the database"""
    try:
        if not supabase_db_manager.is_available():
            return jsonify({'error': 'Database not available'}), 500
        
        formal_report = supabase_db_manager.get_formal_report(formal_report_id)
        if not formal_report:
            return jsonify({'error': 'Formal report not found'}), 404
        
        return jsonify({
            'success': True,
            'formal_report': formal_report
        })
        
    except Exception as e:
        print(f"Error retrieving formal report: {e}")
        return jsonify({'error': 'Failed to retrieve formal report'}), 500

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze every 30th frame of a video with depth maps, segmentation, and classification"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        frame_interval = data.get('frame_interval', 30)  # Default to every 30th frame
        
        print(f"Video analysis request: video_id={video_id}, frame_interval={frame_interval}")
        
        if not video_id:
            return jsonify({'error': 'No video ID provided'}), 400
        
        # Get video metadata
        video_metadata = get_video_metadata(video_id)
        if not video_metadata:
            print(f"Video metadata not found for video_id: {video_id}")
            return jsonify({'error': 'Video not found'}), 404
        
        # Get the actual extracted frames (every 7th frame from original video)
        extracted_frames = video_metadata.get('extracted_frames', 0)
        print(f"Video metadata: extracted_frames={extracted_frames}")
        
        if extracted_frames == 0:
            return jsonify({'error': 'No extracted frames found in video'}), 400
        
        # Calculate frames to process from the extracted frames (every 30th frame from extracted frames)
        # So if we have frames 0-106 (107 frames), we want frames 0, 30, 60, 90
        frames_to_process = list(range(0, extracted_frames, frame_interval))
        total_frames_to_process = len(frames_to_process)
        
        print(f"Processing {total_frames_to_process} frames from {extracted_frames} extracted frames")
        print(f"Frame indices to process: {frames_to_process}")
        
        print(f"Starting video analysis: {total_frames_to_process} frames to process (every {frame_interval}th extracted frame)")
        
        # Initialize results
        analysis_results = {
            'video_id': video_id,
            'total_extracted_frames': extracted_frames,
            'frames_processed': total_frames_to_process,
            'frame_interval': frame_interval,
            'frames': [],
            'processing_times': {
                'total_time': 0,
                'average_per_frame': 0,
                'per_frame_times': []
            },
            'summary': {
                'total_objects': 0,
                'dangerous_objects': 0,
                'safe_objects': 0,
                'unique_labels': set()
            }
        }
        
        start_time = time.time()
        
        # Process each frame
        for i, frame_number in enumerate(frames_to_process):
            frame_start_time = time.time()
            
            try:
                # Get frame info
                frame_info = get_video_frame(video_id, frame_number)
                if not frame_info:
                    print(f"Could not load frame {frame_number}")
                    continue
                
                # Extract the actual file path from the frame info
                frame_path = frame_info.get('path')
                if not frame_path:
                    print(f"No path found for frame {frame_number}")
                    continue
                
                # Process frame with segmentation and depth
                print(f"Processing frame {frame_number}...")
                frame_result = process_single_frame_analysis(frame_path, frame_number)
                
                if frame_result:
                    analysis_results['frames'].append(frame_result)
                    print(f"Frame {frame_number} processed successfully with {frame_result.get('object_count', 0)} objects")
                    
                    # Update summary
                    if 'objects' in frame_result:
                        analysis_results['summary']['total_objects'] += len(frame_result['objects'])
                        for obj in frame_result['objects']:
                            analysis_results['summary']['unique_labels'].add(obj.get('label', 'unknown'))
                            if obj.get('safety', {}).get('is_dangerous', False):
                                analysis_results['summary']['dangerous_objects'] += 1
                            else:
                                analysis_results['summary']['safe_objects'] += 1
                else:
                    print(f"Frame {frame_number} processing failed")
                
                frame_time = time.time() - frame_start_time
                analysis_results['processing_times']['per_frame_times'].append(frame_time)
                
                print(f"Processed frame {frame_number}/{extracted_frames-1} ({i+1}/{total_frames_to_process}) - {frame_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
                continue
        
        # Calculate final timing
        total_time = time.time() - start_time
        analysis_results['processing_times']['total_time'] = total_time
        analysis_results['processing_times']['average_per_frame'] = total_time / total_frames_to_process if total_frames_to_process > 0 else 0
        
        # Convert set to list for JSON serialization
        analysis_results['summary']['unique_labels'] = list(analysis_results['summary']['unique_labels'])
        
        print(f"Video analysis complete: {total_time:.2f}s total, {analysis_results['processing_times']['average_per_frame']:.2f}s per frame")
        
        # Store results for JSON download
        global video_analysis_results
        video_analysis_results[video_id] = analysis_results
        
        # Store initial report in database
        if supabase_db_manager.is_available():
            try:
                # Get video metadata for database storage
                video_metadata = get_video_metadata(video_id)
                video_duration = video_metadata.get('duration_seconds', 0)
                video_captured_at = datetime.now()  # Use current time as captured time
                
                # Create report in database
                report_id = supabase_db_manager.create_report(
                    video_id=video_id,
                    video_duration=video_duration,
                    video_captured_at=video_captured_at,
                    video_device_type="smart glasses"
                )
                
                # Store report_id in analysis results for later use
                analysis_results['report_id'] = report_id
                print(f"Report created in Supabase: {report_id}")
                
            except Exception as e:
                print(f"Error storing report in Supabase: {e}")
                # Don't fail the entire request if database storage fails
                analysis_results['report_id'] = None
        else:
            print("Supabase not available - skipping report creation")
            analysis_results['report_id'] = None
        
        return jsonify(analysis_results)
        
    except Exception as e:
        print(f"Video analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    # Initialize SAM2 model
    sam_loaded = initialize_sam()
    
    # Initialize depth estimation pipeline
    depth_loaded = initialize_depth()
    
    # Initialize CLIP model for classification
    clip_loaded = initialize_clip()
    
    # Initialize Gemini model for safety analysis
    gemini_loaded = initialize_gemini()
    
    if sam_loaded and depth_loaded:
        app.run(debug=True, host='0.0.0.0', port=5004)
    else:
        if not sam_loaded:
            print("Failed to initialize SAM2 model. Please check model paths.")
        if not depth_loaded:
            print("Failed to initialize depth estimation pipeline. Please check transformers installation.")
        if not clip_loaded:
            print("Failed to initialize CLIP model. Classification will be disabled.") 