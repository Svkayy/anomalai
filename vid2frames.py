import cv2
from pathlib import Path

def extract_frames(
    video_path: str,
    out_dir: str,
    every_n_frames: int | None = 30,   # set this OR every_n_seconds
    every_n_seconds: float | None = None,
    png_compression: int = 3           # 0 (none) .. 9 (max compression)
) -> None:
    """
    Save frames as PNGs from `video_path` into `out_dir`.

    Choose one:
      - every_n_frames: save one frame every N frames (e.g., 30)
      - every_n_seconds: save one frame every S seconds (e.g., 1.0)

    If both are given, `every_n_frames` is used.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Determine frame interval
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if every_n_frames is None:
        if not fps:
            raise RuntimeError("Video FPS is unknown; use every_n_frames instead.")
        every_n_frames = max(1, int(round(every_n_seconds * fps)))

    # Read total frames (may be 0 for some files, that’s fine)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Iterate by seeking to frames instead of decoding all frames (faster for long videos)
    frame_idx = 0
    saved = 0

    while True:
        # Seek to desired frame (seeking can be approximate depending on codec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # Timestamp (ms) → seconds, for human-friendly filenames
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_sec = ts_ms / 1000.0 if ts_ms > 0 else (frame_idx / fps if fps else 0)

        # Build filename: zero-padded index + timestamp
        fname = f"frame_{saved:06d}_t{ts_sec:010.3f}.png"
        cv2.imwrite(str(out / fname), frame, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        saved += 1

        # Next target frame
        next_idx = frame_idx + every_n_frames
        if total_frames and next_idx >= total_frames:
            break
        frame_idx = next_idx

    cap.release()
    print(f"Saved {saved} PNGs to {out.resolve()}")

# --- Examples ---
# 1) Every 30 frames (e.g., for a 30 FPS video → 1 image per second)
# extract_frames("input.mp4", "frames_out", every_n_frames=30)

# 2) Every 0.5 seconds (2 images per second, converts using FPS)
# extract_frames("input.mp4", "frames_out", every_n_frames=None, every_n_seconds=0.5)


def main():
    extract_frames("IMG_4227.mov", "frames", every_n_frames=30)


if __name__ == "__main__":
    main()
