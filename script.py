import coremltools as ct
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import time


@dataclass
class Point:
    x: float
    y: float
    label: int  # 0 for background, 1 for foreground


@dataclass
class BoundingBox:
    x1: float  # left
    y1: float  # top
    x2: float  # right
    y2: float  # bottom


class SAM2:
    def __init__(self):
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self.image_embeddings = None
        self.prompt_embeddings = None
        self.input_size = (1024, 1024)  # Fixed size that the model expects
        self.original_image_size = None

    def load_models(
        self, image_encoder_path: str, prompt_encoder_path: str, mask_decoder_path: str
    ):
        """Load all three CoreML models."""
        start_time = time.time()

        self.image_encoder = ct.models.MLModel(image_encoder_path)
        self.prompt_encoder = ct.models.MLModel(prompt_encoder_path)
        self.mask_decoder = ct.models.MLModel(mask_decoder_path)

        initialization_time = time.time() - start_time
        print(f"Models loaded in: {initialization_time:.2f} seconds")

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image to match model input size."""
        image = Image.open(image_path)
        self.original_image_size = image.size

        image = image.convert("RGB")

        # Simple resize without maintaining aspect ratio
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)

        return image

    def get_image_embedding(self, image_path: str):
        """Get image embeddings using the image encoder."""
        if self.image_encoder is None:
            raise ValueError("Models not loaded. Call load_models first.")

        start_time = time.time()

        # Preprocess image and keep as PIL Image
        image = self.preprocess_image(image_path)

        embeddings = self.image_encoder.predict({"image": image})
        self.image_embeddings = embeddings

        duration = time.time() - start_time
        print(f"Image encoding took: {duration:.2f} seconds")

        return embeddings

    def transform_points(
        self, points: List[Point], original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform point coordinates to match model input size and prepare arrays."""
        if len(points) == 0:
            raise ValueError("At least one point is required")

        # Initialize arrays with the correct shape
        coords_array = np.zeros((1, len(points), 2), dtype=np.float32)  # Shape: (1, N, 2)
        labels_array = np.zeros((1, len(points)), dtype=np.int32)  # Shape: (1, N)

        # Transform all points
        for i, point in enumerate(points):
            # Scale coordinates to match the resized image dimensions
            x = point.x * (self.input_size[0] / original_size[0])
            y = point.y * (self.input_size[1] / original_size[1])

            coords_array[0, i] = [x, y]
            labels_array[0, i] = point.label

        return coords_array, labels_array

    def transform_bounding_box(
        self, bbox: BoundingBox, original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform bounding box coordinates to match model input size and prepare arrays."""
        # Scale coordinates to match the resized image dimensions
        x1 = bbox.x1 * (self.input_size[0] / original_size[0])
        y1 = bbox.y1 * (self.input_size[1] / original_size[1])
        x2 = bbox.x2 * (self.input_size[0] / original_size[0])
        y2 = bbox.y2 * (self.input_size[1] / original_size[1])

        # Create two points representing the bounding box: top-left and bottom-right
        # Label 2 = box origin (top-left)
        # Label 3 = box end (bottom-right)
        points = [
            [x1, y1],  # Top-left (box origin)
            [x2, y2]   # Bottom-right (box end)
        ]
        
        # Initialize arrays with the correct shape for 2 points
        coords_array = np.zeros((1, 2, 2), dtype=np.float32)  # Shape: (1, 2, 2)
        labels_array = np.zeros((1, 2), dtype=np.int32)  # Shape: (1, 2)

        # Fill the arrays
        coords_array[0, 0] = [x1, y1]  # Top-left point
        labels_array[0, 0] = 2          # Box origin label
        
        coords_array[0, 1] = [x2, y2]  # Bottom-right point
        labels_array[0, 1] = 3          # Box end label

        return coords_array, labels_array

    def get_prompt_embedding(self, prompts: Union[List[Point], BoundingBox], original_size: Tuple[int, int]):
        """Get prompt embeddings using the prompt encoder."""
        if self.prompt_encoder is None:
            raise ValueError("Models not loaded. Call load_models first.")

        start_time = time.time()

        if isinstance(prompts, list) and all(isinstance(p, Point) for p in prompts):
            # Handle point prompts
            if len(prompts) == 0:
                raise ValueError("At least one point is required")
            points_array, labels_array = self.transform_points(prompts, original_size)
        elif isinstance(prompts, BoundingBox):
            # Handle bounding box prompt
            points_array, labels_array = self.transform_bounding_box(prompts, original_size)
        else:
            raise ValueError("Prompts must be either a list of Points or a BoundingBox")

        # Get prompt embeddings
        self.prompt_embeddings = self.prompt_encoder.predict(
            {"points": points_array, "labels": labels_array}
        )

        duration = time.time() - start_time
        print(f"Prompt encoding took: {duration:.2f} seconds")

        return self.prompt_embeddings

    def get_mask(self, original_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate the mask using the mask decoder."""
        if (
            self.mask_decoder is None
            or self.image_embeddings is None
            or self.prompt_embeddings is None
        ):
            raise ValueError("Models not loaded or embeddings not computed.")

        start_time = time.time()

        try:
            # Get mask prediction
            mask_output = self.mask_decoder.predict(
                {
                    "image_embedding": self.image_embeddings["image_embedding"],
                    "sparse_embedding": self.prompt_embeddings["sparse_embeddings"],
                    "dense_embedding": self.prompt_embeddings["dense_embeddings"],
                    "feats_s0": self.image_embeddings["feats_s0"],
                    "feats_s1": self.image_embeddings["feats_s1"],
                }
            )

            # Get second best mask by score
            # Get best mask (highest score)
            scores = mask_output["scores"]
            print(f"Mask scores: {scores}")
            best_mask_idx = np.argmax(scores)
            mask = mask_output["low_res_masks"][0, best_mask_idx]

            # Resize mask to original image size
            mask = cv2.resize(
                mask,
                (original_size[0], original_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Apply threshold
            mask = (mask > 0).astype(np.float32)

            duration = time.time() - start_time
            print(f"Mask generation took: {duration:.2f} seconds")

            return mask

        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            return None

    def save_mask(self, mask: np.ndarray, output_path: str):
        """Save the mask as a PNG file."""
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_image)

    def apply_mask_to_image(self, image_path, mask):
        image = cv2.imread(image_path)
        mask_binary = mask.astype(np.uint8) * 255
        segmented = cv2.bitwise_and(image, image, mask=mask_binary)

        # Create white background for transparency
        white_background = np.ones_like(image) * 255
        background = cv2.bitwise_and(
            white_background, white_background, mask=~mask_binary
        )
        # Combine segmented image with white background
        final_image = cv2.add(segmented, background)
        return final_image


class PointSelector:
    def __init__(self, image_path, max_points=2):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.display_image = self.image.copy()
        self.points = []
        self.window_name = "Point Selection"
        self.max_points = max_points

    def mouse_callback(
        self,
        event,
        x,
        y,
        _flags,
        _param,
    ):
        if len(self.points) >= self.max_points:
            if not hasattr(self, "max_points_reached"):
                print(
                    "Maximum number of points (2) reached. Press ENTER to continue or 'c' to clear."
                )
                self.max_points_reached = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click for foreground
            self.points.append(Point(x=float(x), y=float(y), label=1))
            self._update_display()
            points_left = self.max_points - len(self.points)
            print(
                f"Added foreground point at ({x}, {y}). {points_left} point{'s' if points_left != 1 else ''} remaining."
            )

        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for background
            self.points.append(Point(x=float(x), y=float(y), label=0))
            self._update_display()
            points_left = self.max_points - len(self.points)
            print(
                f"Added background point at ({x}, {y}). {points_left} point{'s' if points_left != 1 else ''} remaining."
            )

        # Reset the flag if points are cleared
        if len(self.points) < self.max_points and hasattr(self, "max_points_reached"):
            delattr(self, "max_points_reached")

    def _update_display(self):
        self.display_image = self.image.copy()

        # Draw all points
        for point in self.points:
            color = (
                (0, 255, 0) if point.label == 1 else (0, 0, 255)
            )  # Green for foreground, Red for background
            cv2.circle(self.display_image, (int(point.x), int(point.y)), 5, color, -1)

        # Add point count to display
        points_left = self.max_points - len(self.points)
        text = f"Points remaining: {points_left}"
        cv2.putText(
            self.display_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.window_name, self.display_image)

    def select_points(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        cv2.imshow(self.window_name, self.image)

        print("Instructions (Exactly 2 points required):")
        print("- Left click to add foreground point (green)")
        print("- Right click to add background point (red)")
        print("- Press 'c' to clear points")
        print("- Press ENTER when you have added exactly 2 points")
        print("- Press ESC to cancel")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                if len(self.points) != 2:
                    print("Please select exactly 2 points before continuing.")
                    continue
                break
            elif key == 27:  # Escape key
                self.points = []
                break
            elif key == ord("c"):  # Clear points
                self.points = []
                self._update_display()
                print("Cleared all points. Please select exactly 2 points.")

        cv2.destroyAllWindows()
        return self.points


class BoundingBoxSelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.display_image = self.image.copy()
        self.bbox = None
        self.window_name = "Bounding Box Selection"
        self.start_point = None
        self.drawing = False

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
            self.display_image = self.image.copy()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.display_image = self.image.copy()
            cv2.rectangle(
                self.display_image, self.start_point, (x, y), (0, 255, 0), 2
            )

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            
            # Ensure coordinates are in correct order (x1,y1 is top-left, x2,y2 is bottom-right)
            x1, x2 = min(self.start_point[0], end_point[0]), max(self.start_point[0], end_point[0])
            y1, y2 = min(self.start_point[1], end_point[1]), max(self.start_point[1], end_point[1])
            
            self.bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            
            # Draw final rectangle
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.display_image)
            
            print(f"Bounding box selected: ({x1}, {y1}) to ({x2}, {y2})")

    def _update_display(self):
        self.display_image = self.image.copy()
        if self.bbox:
            cv2.rectangle(
                self.display_image,
                (int(self.bbox.x1), int(self.bbox.y1)),
                (int(self.bbox.x2), int(self.bbox.y2)),
                (0, 255, 0),
                2,
            )
        cv2.imshow(self.window_name, self.display_image)

    def select_bounding_box(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        cv2.imshow(self.window_name, self.image)

        print("Instructions:")
        print("- Click and drag to draw a bounding box")
        print("- Press ENTER to confirm the bounding box")
        print("- Press 'c' to clear and redraw")
        print("- Press ESC to cancel")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                if self.bbox is None:
                    print("Please draw a bounding box before continuing.")
                    continue
                break
            elif key == 27:  # Escape key
                self.bbox = None
                break
            elif key == ord("c"):  # Clear bounding box
                self.bbox = None
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
                print("Cleared bounding box. Please draw a new one.")

        cv2.destroyAllWindows()
        return self.bbox


def main():
    try:
        sam = SAM2()

        # Set to the paths of the CoreML models
        sam.load_models(
            # image_encoder_path="./models/SAM2_1LargeImageEncoderFLOAT16.mlpackage",
            # prompt_encoder_path="./models/SAM2_1LargePromptEncoderFLOAT16.mlpackage",
            # mask_decoder_path="./models/SAM2_1LargeMaskDecoderFLOAT16.mlpackage",
            image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
            prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
            mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
        )

        # Set to the path of the image you want to process
        image_path = "./poster.jpg"

        # Choose prompt type
        print("Choose prompt type:")
        print("1. Point prompts (2 points)")
        print("2. Bounding box prompt")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Use point prompts
            point_selector = PointSelector(image_path, max_points=2)
            prompts = point_selector.select_points()
            
            if len(prompts) != 2:
                print("Exactly 2 points are required. Exiting.")
                return
                
        elif choice == "2":
            # Use bounding box prompt
            bbox_selector = BoundingBoxSelector(image_path)
            prompts = bbox_selector.select_bounding_box()
            
            if prompts is None:
                print("No bounding box selected. Exiting.")
                return
        else:
            print("Invalid choice. Exiting.")
            return

        print(f"Selected prompts: {prompts}")

        sam.get_image_embedding(image_path)
        original_size = Image.open(image_path).size
        sam.prompt_embeddings = sam.get_prompt_embedding(prompts, original_size)
        mask = sam.get_mask(original_size)

        if mask is not None:
            # Save the mask
            sam.save_mask(mask, "output_mask.png")

            # Save segmented image
            segmented_image = sam.apply_mask_to_image(image_path, mask)
            cv2.imwrite("output_segmented.png", segmented_image)

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
