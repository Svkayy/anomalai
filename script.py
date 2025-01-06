import coremltools as ct
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class Point:
    x: float
    y: float
    label: int  # 0 for background, 1 for foreground


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
        if len(points) != 2:
            raise ValueError("Exactly 2 points are required for this model")

        # Initialize arrays with the correct shape
        coords_array = np.zeros((1, 2, 2), dtype=np.float32)  # Shape: (1, 2, 2)
        labels_array = np.zeros((1, 2), dtype=np.int32)  # Shape: (1, 2)

        # Transform both points
        for i, point in enumerate(points):
            # Scale coordinates to match the resized image dimensions
            x = point.x * (self.input_size[0] / original_size[0])
            y = point.y * (self.input_size[1] / original_size[1])

            coords_array[0, i] = [x, y]
            labels_array[0, i] = point.label

        return coords_array, labels_array

    def get_prompt_embedding(self, points: List[Point], original_size: Tuple[int, int]):
        """Get prompt embeddings using the prompt encoder."""
        if self.prompt_encoder is None:
            raise ValueError("Models not loaded. Call load_models first.")

        if len(points) != 2:
            raise ValueError("Exactly 2 points are required for this model")

        start_time = time.time()

        # Transform points and get properly shaped arrays
        points_array, labels_array = self.transform_points(points, original_size)

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


def main():
    try:
        sam = SAM2()

        # Set to the paths of the CoreML models
        sam.load_models(
            image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
            prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
            mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
        )

        # Set to the path of the image you want to process
        image_path = "./potplants.png"

        point_selector = PointSelector(image_path, max_points=2)
        points = point_selector.select_points()

        if len(points) != 2:
            print("Exactly 2 points are required. Exiting.")
            return

        print(f"Selected {len(points)} points")

        sam.get_image_embedding(image_path)
        original_size = Image.open(image_path).size
        sam.prompt_embeddings = sam.get_prompt_embedding(points, original_size)
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
