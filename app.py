from flask import Flask, render_template, request, jsonify, send_file
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SAM2 model (global variable)
sam_model = None

# Initialize depth estimation pipeline (global variable)
depth_pipe = None

# Initialize CLIP model for classification (global variables)
clip_model = None
clip_preprocess = None
clip_device = None

# Parallel processing configuration
parallel_config = {
    'max_workers': 4,
    'batch_size': 32,
    'target_points': 256,
    'max_masks': 50
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
            
            # Classify masks using CLIP
            mask_labels = []
            if clip_model is not None and len(kept_masks) > 0:
                # Define common object labels for classification
                LABELS = [
                    "person", "car", "truck", "bus", "motorcycle", "bicycle",
                    "dog", "cat", "bird", "horse", "cow", "sheep",
                    "tree", "grass", "sky", "water", "road", "building",
                    "house", "window", "door", "fence", "sign", "traffic light",
                    "chair", "table", "bed", "sofa", "lamp", "book",
                    "phone", "laptop", "keyboard", "mouse", "monitor", "television",
                    "bottle", "cup", "bowl", "plate", "fork", "knife", "spoon",
                    "banana", "apple", "orange", "broccoli", "carrot", "pizza",
                    "donut", "cake", "sandwich", "hot dog", "hamburger", "french fries"
                ]
                
                # Prepare text features once
                text_features = prepare_text_features(LABELS)
                
                if text_features is not None:
                    print(f"Classifying {len(kept_masks)} masks...")
                    for i, mask in enumerate(kept_masks):
                        label, confidence = classify_mask(filepath, mask, LABELS, text_features)
                        mask_labels.append({
                            "label": label, 
                            "confidence": round(confidence, 3),
                            "mask_index": i
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

            pil_image = Image.fromarray(final_image, 'RGBA')
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

@app.route('/depth/<file_id>', methods=['GET'])
def get_depth(file_id):
    """Generate depth map for uploaded image using Depth Anything"""
    if depth_pipe is None:
        return jsonify({'error': 'Depth estimation pipeline not loaded'}), 500
        
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
        'target_points': data.get('target_points', 256),
        'max_masks': data.get('max_masks', 50)
    }
    
    return jsonify({
        'success': True,
        'config': parallel_config
    })

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
    
    if sam_loaded and depth_loaded:
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        if not sam_loaded:
            print("Failed to initialize SAM2 model. Please check model paths.")
        if not depth_loaded:
            print("Failed to initialize depth estimation pipeline. Please check transformers installation.")
        if not clip_loaded:
            print("Failed to initialize CLIP model. Classification will be disabled.") 