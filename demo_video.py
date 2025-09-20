#!/usr/bin/env python3
"""
Demo script for video frame-by-frame analysis
"""

import os
import sys
from video_processor import process_video_upload, get_video_frame, get_video_metadata

def demo_video_processing():
    """Demonstrate video processing functionality"""
    
    # Check if test video exists
    test_video = "test_video.mp4"
    if not os.path.exists(test_video):
        print(f"Test video {test_video} not found. Please create it first.")
        return
    
    print("🎬 Video Frame-by-Frame Analysis Demo")
    print("=" * 50)
    
    # Process video
    print(f"Processing video: {test_video}")
    frames, metadata = process_video_upload(test_video, max_frames=30)
    
    if not frames:
        print("❌ Failed to extract frames from video")
        return
    
    print(f"✅ Successfully extracted {len(frames)} frames")
    print(f"📊 Video Info:")
    print(f"   - Duration: {metadata['duration']:.2f} seconds")
    print(f"   - FPS: {metadata['fps']:.2f}")
    print(f"   - Total Frames: {metadata['total_frames']}")
    print(f"   - Extracted Frames: {metadata['extracted_frames']}")
    print(f"   - Frame Interval: {metadata['frame_interval']}")
    
    # Show frame information
    print(f"\n🎞️  Frame Information:")
    for i, frame in enumerate(frames[:5]):  # Show first 5 frames
        print(f"   Frame {frame['frame_number']}: {frame['timestamp']:.2f}s")
    
    if len(frames) > 5:
        print(f"   ... and {len(frames) - 5} more frames")
    
    # Test frame retrieval
    print(f"\n🔍 Testing frame retrieval:")
    test_frame = get_video_frame(metadata['video_id'], 0)
    if test_frame:
        print(f"   ✅ Frame 0 retrieved successfully")
        print(f"   📐 Dimensions: {test_frame['width']}x{test_frame['height']}")
    else:
        print(f"   ❌ Failed to retrieve frame 0")
    
    print(f"\n🚀 Ready for web interface!")
    print(f"   - Upload the video through the web interface")
    print(f"   - Use the slider to navigate between frames")
    print(f"   - Click 'Process Selection' to segment each frame")
    print(f"   - Labels will be overlaid directly on each frame")

if __name__ == "__main__":
    demo_video_processing()
