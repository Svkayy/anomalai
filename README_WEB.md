# SAM2 Web Application

A web-based interface for SAM2 (Segment Anything Model 2) image segmentation using CoreML models.

## Features

- 🖼️ **Image Upload**: Drag and drop or browse to upload images
- 🎯 **Point Selection**: Click to add foreground (green) and background (red) points
- 📦 **Bounding Box**: Draw bounding boxes by clicking and dragging
- 🎨 **Real-time Preview**: See your selections on the original image
- ✨ **Modern UI**: Beautiful, responsive interface
- 🚀 **Fast Processing**: Powered by CoreML for efficient inference

## Prerequisites

- Python 3.8+
- CoreML-compatible device (macOS with Apple Silicon recommended)
- SAM2 CoreML models (see model setup below)

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download SAM2 CoreML models**:
   - Navigate to the `coreml-sam2.1-small/` or `coreml-sam2.1-large/` directory
   - Follow the instructions in the respective README files to download the models
   - Place the models in the `models/` directory with the following structure:
     ```
     models/
     ├── SAM2_1SmallImageEncoderFLOAT16.mlpackage
     ├── SAM2_1SmallPromptEncoderFLOAT16.mlpackage
     └── SAM2_1SmallMaskDecoderFLOAT16.mlpackage
     ```

## Usage

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload an image**:
   - Drag and drop an image onto the upload area, or
   - Click "Choose Image" to browse for a file

4. **Choose selection method**:
   - **Point Selection**: Click to add exactly 2 points (left click for foreground, right click for background)
   - **Bounding Box**: Click and drag to draw a bounding box around the object

5. **Segment the image**:
   - Click "🎯 Segment Image" to process the selection
   - View the segmented result in the right panel

## How it Works

1. **Image Processing**: The uploaded image is processed and stored on the server
2. **Selection**: Users can select points or draw bounding boxes on the image
3. **SAM2 Processing**: The selections are sent to the SAM2 model for segmentation
4. **Result Display**: The segmented image is returned and displayed to the user

## File Structure

```
sam2-coreml-python/
├── app.py                 # Flask web application
├── script.py             # SAM2 CoreML implementation
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface template
├── static/              # Static assets (if any)
├── uploads/             # Temporary uploaded images
├── models/              # SAM2 CoreML models
└── README_WEB.md       # This file
```

## Troubleshooting

### Model Loading Issues
- Ensure the CoreML models are properly downloaded and placed in the `models/` directory
- Check that the model paths in `app.py` match your actual model filenames
- Verify that you're running on a CoreML-compatible device

### Performance Issues
- The first run may be slower as models are loaded into memory
- Large images will take longer to process
- Consider using the small model variant for faster processing

### Browser Issues
- Ensure JavaScript is enabled in your browser
- Try refreshing the page if the interface doesn't respond
- Check the browser console for any JavaScript errors

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload image file
- `POST /segment`: Process image segmentation with prompts

## Development

To modify the application:

1. **Frontend**: Edit `templates/index.html` for UI changes
2. **Backend**: Edit `app.py` for server-side logic
3. **SAM2 Logic**: Edit `script.py` for model-related changes

## License

This project uses the SAM2 model which is subject to its own license terms. Please refer to the original SAM2 repository for licensing information. 