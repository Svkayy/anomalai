# Video Frame-by-Frame Analysis

This document describes the video analysis functionality added to the SAM2 segmentation application.

## Overview

The application now supports video upload and frame-by-frame analysis with an iMovie-style interface for navigating through video frames and applying segmentation with overlaid labels.

## Features

### 🎬 Video Upload
- Support for multiple video formats: `.mov`, `.mp4`, `.avi`, `.mkv`, `.webm`
- Automatic frame extraction (up to 100 frames by default)
- Video metadata display (duration, FPS, total frames, etc.)

### 🎚️ iMovie-Style Controls
- **Frame Slider**: Navigate through video frames with a smooth slider
- **Play/Pause**: Play through frames at video FPS or pause for detailed analysis
- **Previous/Next**: Step through frames one by one
- **Frame Counter**: Shows current frame number and total frames
- **Timestamp**: Displays current time position in the video

### 🎯 Frame Segmentation
- Apply the same segmentation techniques to individual video frames
- Grid segmentation with automatic object detection
- Overlaid labels showing object names and confidence scores
- Real-time processing of selected frames

## Usage

### 1. Upload Video
1. Click "Choose File" and select a video file
2. The video will be processed and frames extracted automatically
3. Video controls will appear below the upload section

### 2. Navigate Frames
- Use the slider to jump to any frame
- Click play/pause to automatically advance through frames
- Use previous/next buttons for precise frame control

### 3. Analyze Frames
1. Select a frame using the controls
2. Choose "Grid Segmentation" mode
3. Click "Process Selection" to segment the current frame
4. View the segmented result with overlaid labels

## Technical Implementation

### Backend Components

#### Video Processor (`video_processor.py`)
- Extracts frames from uploaded videos using OpenCV
- Saves frames as PNG images for processing
- Generates metadata including timestamps and frame information
- Supports configurable frame sampling rates

#### API Endpoints
- `POST /upload_video`: Upload and process video files
- `GET /video/<video_id>/frame/<frame_number>`: Retrieve specific frames
- `GET /video/<video_id>/metadata`: Get video metadata
- `POST /segment_video_frame`: Segment a specific video frame

### Frontend Components

#### Video Controls UI
- iMovie-style slider with custom styling
- Play/pause functionality with frame rate timing
- Real-time frame information display
- Responsive design for different screen sizes

#### JavaScript Functions
- `handleVideoFile()`: Process video uploads
- `loadVideoFrame()`: Load and display specific frames
- `togglePlayPause()`: Control video playback
- `segmentVideoFrame()`: Apply segmentation to current frame

## File Structure

```
sam2-coreml-python/
├── video_processor.py          # Video processing module
├── app.py                      # Modified Flask app with video routes
├── templates/index.html        # Updated UI with video controls
├── demo_video.py              # Demo script
├── test_video.mp4             # Test video (generated)
└── uploads/
    └── video_<id>_frames/      # Extracted frame directories
        ├── frame_0000.png
        ├── frame_0001.png
        └── metadata.json
```

## Configuration

### Video Processing Settings
- Maximum frames extracted: 100 (configurable)
- Frame sampling: Automatic based on video length
- Supported formats: MOV, MP4, AVI, MKV, WebM
- Frame format: PNG for quality preservation

### Performance Considerations
- Frames are extracted once and cached
- Large videos are sampled to avoid memory issues
- Frame loading is optimized for web display
- Segmentation is applied on-demand per frame

## Example Usage

```python
from video_processor import process_video_upload, get_video_frame

# Process a video
frames, metadata = process_video_upload('video.mp4', max_frames=50)

# Get a specific frame
frame_info = get_video_frame(metadata['video_id'], 0)
```

## Browser Compatibility

- Modern browsers with ES6 support
- Canvas API for image display
- Range input support for sliders
- File API for video uploads

## Future Enhancements

- Batch processing of multiple frames
- Video export with segmentation overlays
- Frame interpolation for smoother playback
- Advanced video analysis features
- Real-time video streaming support

## Troubleshooting

### Common Issues
1. **Video not processing**: Check file format and size
2. **Frames not loading**: Verify video was processed successfully
3. **Segmentation slow**: Large videos may take time to process
4. **Memory issues**: Reduce max_frames parameter for large videos

### Debug Information
- Check browser console for JavaScript errors
- Monitor server logs for processing errors
- Verify video file integrity and format support
