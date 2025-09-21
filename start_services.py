#!/usr/bin/env python3
"""
Startup script for both Flask app and FastAPI safety classifier
"""

import subprocess
import time
import sys
import os
from threading import Thread

def run_flask_app():
    """Run the Flask application"""
    print("Starting Flask application...")
    os.system("python app.py")

def run_safety_api():
    """Run the FastAPI safety classifier"""
    print("Starting Safety Classification API...")
    os.system("python safety_api.py")

def main():
    print("🚀 Starting SAM2-CoreML-Python Services")
    print("=" * 50)
    
    # Start Flask app in a separate thread
    flask_thread = Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Wait a moment for Flask to start
    time.sleep(3)
    
    # Start FastAPI safety service in a separate thread
    safety_thread = Thread(target=run_safety_api, daemon=True)
    safety_thread.start()
    
    print("\n✅ Services started successfully!")
    print("\n📱 Main Application: http://localhost:5000")
    print("🔒 Safety API: http://localhost:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()
