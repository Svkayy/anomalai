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
        self.input_size = (1024, 1024)  # Fixed size that the model seems to expect
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
    ) -> np.ndarray:
        """Transform point coordinates to match model input size."""
        transformed_points = []
        for point in points:
            # Simple scaling to match the resized image dimensions
            transformed_x = point.x * (self.input_size[0] / original_size[0])
            transformed_y = point.y * (self.input_size[1] / original_size[1])

            transformed_points.append([transformed_x, transformed_y])

        return np.array(transformed_points)

    def get_prompt_embedding(self, points: List[Point], original_size: Tuple[int, int]):
        """Get prompt embeddings using the prompt encoder."""
        if self.prompt_encoder is None:
            raise ValueError("Models not loaded. Call load_models first.")

        start_time = time.time()

        # Transform points to match model input size
        transformed_points = self.transform_points(points, original_size)

        # Prepare inputs for prompt encoder
        points_array = transformed_points.reshape(1, -1, 2).astype(np.float32)
        labels_array = np.array([[p.label for p in points]], dtype=np.int32)

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
        print(f"Scores: {scores}")
        best_mask_idx = np.argmax(scores)
        mask = mask_output["low_res_masks"][0, best_mask_idx]

        # Resize mask to original image size
        mask = cv2.resize(
            mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR
        )

        # Apply threshold
        mask = (mask > 0).astype(np.float32)

        duration = time.time() - start_time
        print(f"Mask generation took: {duration:.2f} seconds")

        return mask

    def save_mask(self, mask: np.ndarray, output_path: str):
        """Save the mask as a PNG file."""
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_image)


def main():
    sam = SAM2()

    # Set to the paths of the CoreML models
    sam.load_models(
        image_encoder_path="./models/SAM2_1SmallImageEncoderFLOAT16.mlpackage",
        prompt_encoder_path="./models/SAM2_1SmallPromptEncoderFLOAT16.mlpackage",
        mask_decoder_path="./models/SAM2_1SmallMaskDecoderFLOAT16.mlpackage",
    )

    # Set to the path of the image you want to process
    image_path = "./cat.png"

    sam.get_image_embedding(image_path)

    # Define points (example: one foreground point)
    points = [Point(x=100, y=100, label=1)]  # Coordinates in original image space

    original_size = Image.open(image_path).size

    sam.prompt_embeddings = sam.get_prompt_embedding(points, original_size)

    mask = sam.get_mask(original_size)

    if mask is not None:
        sam.save_mask(mask, "output_mask.png")


if __name__ == "__main__":
    main()
