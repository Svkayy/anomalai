import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

# Run pipeline
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cpu")
result = pipe(Image.open("landscape.png"))

# Get the raw depth map (numpy array)
depth = np.array(result["depth"])

# Normalize to [0, 255]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth = depth.astype("uint8")

# Apply inferno colormap (purple/yellow/orange)
colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

# Save
cv2.imwrite("depth_colored.png", colored)
print("Saved to depth_colored.png")
