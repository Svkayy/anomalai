#!/usr/bin/env python3
"""
Video processing module for frame-by-frame analysis
"""

import cv2
import imageio
import numpy as np
from PIL import Image
import os
import uuid
from typing import List, Tuple, Optional
import json

class VideoProcessor:
    """Handles video processing and frame extraction"""
    
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def extract_frames(self, video_path: str, max_frames: int = 100, video_id: str = None) -> List[dict]:
        """
        Extract frames from video and save them as images
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame information dictionaries
        """
        try:
            # Read video using OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            # Use fixed frame interval of 7 for consistent sampling
            frame_interval = 4
            # Calculate how many frames we'll actually extract
            actual_frames = total_frames // frame_interval
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            # Use provided video ID or generate new one
            if video_id is None:
                video_id = str(uuid.uuid4())
            video_frames_dir = os.path.join(self.upload_folder, f"video_{video_id}_frames")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            while cap.isOpened() and extracted_count < actual_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame for higher quality processing (max width 2560)
                    height, width = frame_rgb.shape[:2]
                    if width > 2560:  # Increased from 1920 to 2560 for better quality
                        scale = 2560 / width
                        new_width = 2560
                        new_height = int(height * scale)
                        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Save frame with higher quality compression
                    frame_filename = f"frame_{extracted_count:04d}.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    pil_image.save(frame_path, 'JPEG', quality=95, optimize=True)  # Increased from 85 to 95
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    frames.append({
                        'frame_number': extracted_count,
                        'original_frame_number': frame_count,
                        'timestamp': timestamp,
                        'filename': frame_filename,
                        'path': frame_path,
                        'width': frame.shape[1],
                        'height': frame.shape[0]
                    })
                    
                    extracted_count += 1
                    
                    # Print progress
                    progress = (extracted_count / actual_frames) * 100
                    print(f"Extracting frames: {extracted_count}/{actual_frames} ({progress:.1f}%)")
                
                frame_count += 1
            
            cap.release()
            
            # Save video metadata
            metadata = {
                'video_id': video_id,
                'original_path': video_path,
                'total_frames': total_frames,
                'extracted_frames': len(frames),
                'fps': fps,
                'duration': duration,
                'frame_interval': frame_interval,
                'frames_dir': video_frames_dir
            }
            
            metadata_path = os.path.join(video_frames_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Extracted {len(frames)} frames from video")
            return frames, metadata
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return [], {}
    
    def get_frame_info(self, video_id: str, frame_number: int) -> Optional[dict]:
        """Get information about a specific frame"""
        try:
            video_frames_dir = os.path.join(self.upload_folder, f"video_{video_id}_frames")
            metadata_path = os.path.join(video_frames_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            frame_filename = f"frame_{frame_number:04d}.jpg"
            frame_path = os.path.join(video_frames_dir, frame_filename)
            
            if not os.path.exists(frame_path):
                return None
            
            # Get frame dimensions
            with Image.open(frame_path) as img:
                width, height = img.size
            
            return {
                'frame_number': frame_number,
                'filename': frame_filename,
                'path': frame_path,
                'width': width,
                'height': height,
                'video_metadata': metadata
            }
            
        except Exception as e:
            print(f"Error getting frame info: {e}")
            return None
    
    def get_video_metadata(self, video_id: str) -> Optional[dict]:
        """Get video metadata"""
        try:
            video_frames_dir = os.path.join(self.upload_folder, f"video_{video_id}_frames")
            metadata_path = os.path.join(video_frames_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error getting video metadata: {e}")
            return None
    
    def cleanup_video_frames(self, video_id: str):
        """Clean up extracted frames for a video"""
        try:
            video_frames_dir = os.path.join(self.upload_folder, f"video_{video_id}_frames")
            if os.path.exists(video_frames_dir):
                import shutil
                shutil.rmtree(video_frames_dir)
                print(f"Cleaned up frames for video {video_id}")
        except Exception as e:
            print(f"Error cleaning up video frames: {e}")

# Global video processor instance
video_processor = VideoProcessor()

def process_video_upload(video_path: str, max_frames: int = 100, video_id: str = None) -> Tuple[List[dict], dict]:
    """Process uploaded video and extract frames"""
    return video_processor.extract_frames(video_path, max_frames, video_id)

def get_video_frame(video_id: str, frame_number: int) -> Optional[dict]:
    """Get specific frame information"""
    return video_processor.get_frame_info(video_id, frame_number)

def get_video_metadata(video_id: str) -> Optional[dict]:
    """Get video metadata"""
    return video_processor.get_video_metadata(video_id)

if __name__ == "__main__":
    # Test video processing
    processor = VideoProcessor()
    
    # Test with a sample video (if it exists)
    test_video = "IMG_4227.MOV"
    if os.path.exists(test_video):
        print(f"Processing video: {test_video}")
        frames, metadata = processor.extract_frames(test_video, max_frames=50)
        print(f"Extracted {len(frames)} frames")
        print(f"Video metadata: {metadata}")
    else:
        print(f"Test video {test_video} not found")
