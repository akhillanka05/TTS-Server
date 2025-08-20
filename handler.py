#!/usr/bin/env python3
"""
Simplified RunPod Serverless Handler - Lazy initialization
"""

import runpod
import os
import sys
import tempfile
import base64
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/app')

# Global engine instance (initialized on first request, not startup)
engine = None

def initialize_engine():
    """Initialize the TTS engine only when needed (lazy loading)"""
    global engine
    
    if engine is not None:
        return True
        
    try:
        print("🔄 Initializing TTS Engine on first request...")
        
        # Import your existing components
        from engine import ChatterboxEngine
        from config import config_manager
        
        # Load configuration
        config = config_manager.get_config()
        
        # Force CUDA for RunPod GPU instances
        if 'tts_engine' in config:
            config['tts_engine']['device'] = 'cuda'
        
        # Initialize the engine
        engine = ChatterboxEngine(config)
        print("✅ TTS Engine initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize TTS engine: {str(e)}")
        print(traceback.format_exc())
        return False

def handler(job):
    """
    Main RunPod serverless handler
    Only initializes engine when first request comes in
    """
    
    try:
        # Parse job input
        job_input = job.get("input", {})
        request_type = job_input.get("type", "tts")
        
        print(f"📥 Received request type: {request_type}")
        
        # Handle status check without initializing engine
        if request_type == "status":
            return {
                "success": True,
                "message": "Worker is healthy and ready",
                "engine_initialized": engine is not None,
                "python_version": sys.version,
                "working_directory": os.getcwd()
            }
        
        # For TTS requests, initialize engine if needed
        if request_type == "tts":
            if not initialize_engine():
                return {"error": "Failed to initialize TTS engine"}
            
            # Process TTS request (simplified for now)
            text = job_input.get("text", "")
            if not text:
                return {"error": "Text is required"}
            
            return {
                "success": True,
                "message": f"TTS request received for text: '{text[:50]}...'",
                "note": "This is a simplified response. Full TTS processing will be implemented once worker startup is stable."
            }
        
        elif request_type == "get_voices":
            return {
                "success": True,
                "voices": [],
                "count": 0,
                "message": "Voice listing will be implemented once engine is stable"
            }
        
        else:
            return {
                "error": f"Unknown request type: {request_type}",
                "supported_types": ["tts", "get_voices", "status"]
            }
    
    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Handler error: {str(e)}"}

# Entry point for RunPod serverless
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless TTS Worker (Simplified)")
    print("Engine will be initialized on first TTS request")
    
    # Start the RunPod serverless worker immediately
    # No engine initialization during startup
    runpod.serverless.start({"handler": handler})