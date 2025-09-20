#!/usr/bin/env python3
"""
Test script to verify OCR functionality and identify issues
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_imports():
    """Test if required packages can be imported"""
    try:
        import pytesseract
        print("✓ pytesseract imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import pytesseract: {e}")
        return False

def test_tesseract_binary():
    """Test if Tesseract binary is available"""
    try:
        import pytesseract
        # This will raise an error if Tesseract is not installed
        pytesseract.get_tesseract_version()
        print("✓ Tesseract binary found and working")
        return True
    except Exception as e:
        print(f"✗ Tesseract binary not found or not working: {e}")
        print("  You need to install Tesseract OCR on your system:")
        print("  - macOS: brew install tesseract")
        print("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def test_ocr_on_sample_image():
    """Test OCR on a sample image from uploads folder"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"✗ Uploads directory not found: {uploads_dir}")
        return False
    
    # Find a PNG file to test with
    png_files = [f for f in os.listdir(uploads_dir) if f.endswith('.png')]
    if not png_files:
        print("✗ No PNG files found in uploads directory")
        return False
    
    test_file = png_files[0]
    filepath = os.path.join(uploads_dir, test_file)
    print(f"Testing OCR on: {test_file}")
    
    try:
        # Read the image
        image = cv2.imread(filepath)
        if image is None:
            print(f"✗ Failed to read image: {filepath}")
            return False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Test OCR
        import pytesseract
        text = pytesseract.image_to_string(pil_image)
        
        print(f"✓ OCR successful! Extracted text length: {len(text)}")
        if text.strip():
            print(f"Sample text: {text[:100]}...")
        else:
            print("No text detected (this might be normal for some images)")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing OCR functionality...\n")
    
    # Test 1: Package imports
    if not test_imports():
        print("\nFix: Install pytesseract package")
        print("pip install pytesseract")
        return
    
    # Test 2: Tesseract binary
    if not test_tesseract_binary():
        return
    
    # Test 3: Actual OCR functionality
    test_ocr_on_sample_image()
    
    print("\nOCR setup complete!")

if __name__ == "__main__":
    main() 