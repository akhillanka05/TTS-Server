#!/usr/bin/env python3
"""
RunPod Serverless Handler for Chatterbox TTS Server
This file is the main entry point for RunPod serverless deployment
"""

import runpod
import os
import sys
import tempfile
import base64
import traceback
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/app')

# Import your existing TTS components
try:
    from engine import ChatterboxEngine
    from config import config_manager
    print("✅ Successfully imported TTS components")
except ImportError as e:
    print(f"❌ Failed to import TTS components: {e}")
    print("Make sure engine.py and config.py are in the container")
    sys.exit(1)

# Global engine instance (initialized once per worker)
engine = None

def initialize_engine():
    """Initialize the TTS engine once when the worker starts"""
    global engine
    
    if engine is not None:
        print("Engine already initialized")
        return True
        
    try:
        print("🔄 Initializing TTS Engine...")
        
        # Load configuration
        config = config_manager.get_config()
        print(f"Config loaded: {type(config)}")
        
        # Ensure GPU usage for RunPod
        if 'tts_engine' in config:
            config['tts_engine']['device'] = 'cuda'
            print("Set device to CUDA for RunPod GPU")
        
        # Initialize the engine with your existing class
        engine = ChatterboxEngine(config)
        print("✅ TTS Engine initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize TTS engine: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return False

def process_tts_request(input_data):
    """
    Process a TTS generation request
    
    Expected input:
    {
        "text": "Text to synthesize",
        "voice_mode": "predefined" or "clone",
        "predefined_voice_id": "voice_name.wav" (if predefined),
        "reference_audio_base64": "..." (if clone),
        "temperature": 0.7,
        "exaggeration": 1.0,
        "cfg_weight": 3.0,
        "seed": -1,
        "speed_factor": 1.0,
        "language": "en",
        "split_text": true,
        "chunk_size": 120,
        "output_format": "wav"
    }
    """
    
    # Validate required fields
    text = input_data.get("text", "").strip()
    if not text:
        return {"error": "Text is required and cannot be empty"}
    
    # Extract parameters with defaults
    voice_mode = input_data.get("voice_mode", "predefined")
    output_format = input_data.get("output_format", "wav")
    
    # Generation parameters
    temperature = float(input_data.get("temperature", 0.7))
    exaggeration = float(input_data.get("exaggeration", 1.0))
    cfg_weight = float(input_data.get("cfg_weight", 3.0))
    seed = int(input_data.get("seed", -1))
    speed_factor = float(input_data.get("speed_factor", 1.0))
    language = input_data.get("language", "en")
    split_text = input_data.get("split_text", True)
    chunk_size = int(input_data.get("chunk_size", 120))
    
    print(f"🎯 Processing TTS request:")
    print(f"   Text length: {len(text)} characters")
    print(f"   Voice mode: {voice_mode}")
    print(f"   Parameters: temp={temperature}, seed={seed}")
    
    # Handle voice selection
    voice_audio_path = None
    temp_files_cleanup = []
    
    try:
        if voice_mode == "predefined":
            predefined_voice_id = input_data.get("predefined_voice_id", "Alice.wav")
            
            # Ensure .wav extension
            if not predefined_voice_id.endswith('.wav'):
                predefined_voice_id += '.wav'
            
            voice_audio_path = os.path.join("/app/voices", predefined_voice_id)
            
            if not os.path.exists(voice_audio_path):
                # List available voices for debugging
                voices_dir = Path("/app/voices")
                available_voices = []
                if voices_dir.exists():
                    available_voices = [f.name for f in voices_dir.glob("*.wav")]
                
                return {
                    "error": f"Predefined voice '{predefined_voice_id}' not found",
                    "available_voices": available_voices
                }
            
            print(f"✅ Using predefined voice: {predefined_voice_id}")
        
        elif voice_mode == "clone":
            reference_audio_base64 = input_data.get("reference_audio_base64")
            if not reference_audio_base64:
                return {"error": "reference_audio_base64 is required for voice cloning"}
            
            try:
                # Decode base64 audio
                audio_data = base64.b64decode(reference_audio_base64)
                
                # Create temporary file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_data)
                temp_audio.close()
                
                voice_audio_path = temp_audio.name
                temp_files_cleanup.append(voice_audio_path)
                
                print(f"✅ Created temporary reference audio: {voice_audio_path}")
                
            except Exception as e:
                return {"error": f"Failed to decode reference audio: {str(e)}"}
        
        else:
            return {"error": f"Invalid voice_mode: {voice_mode}. Use 'predefined' or 'clone'"}
        
        # Generate speech using your existing engine
        try:
            print("🎵 Generating speech...")
            
            # Call your existing engine method
            audio_result = engine.generate_speech(
                text=text,
                voice_audio_path=voice_audio_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                seed=seed,
                speed_factor=speed_factor,
                language=language,
                split_text=split_text,
                chunk_size=chunk_size,
                output_format=output_format
            )
            
            # Handle different return types from your engine
            if isinstance(audio_result, (str, Path)):
                # Engine returned a file path
                audio_file_path = str(audio_result)
                if os.path.exists(audio_file_path):
                    with open(audio_file_path, 'rb') as f:
                        audio_bytes = f.read()
                    # Clean up the generated file
                    os.unlink(audio_file_path)
                else:
                    raise FileNotFoundError(f"Generated audio file not found: {audio_file_path}")
                    
            elif isinstance(audio_result, bytes):
                # Engine returned raw bytes
                audio_bytes = audio_result
            else:
                raise TypeError(f"Unexpected return type from engine: {type(audio_result)}")
            
            # Encode as base64 for JSON response
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"✅ Speech generated successfully!")
            print(f"   Audio size: {len(audio_bytes)} bytes")
            print(f"   Base64 size: {len(audio_base64)} characters")
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "metadata": {
                    "format": output_format,
                    "text_length": len(text),
                    "voice_mode": voice_mode,
                    "audio_size_bytes": len(audio_bytes),
                    "generation_params": {
                        "temperature": temperature,
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "seed": seed,
                        "speed_factor": speed_factor,
                        "language": language,
                        "split_text": split_text,
                        "chunk_size": chunk_size
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ TTS generation failed: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return {"error": f"TTS generation failed: {str(e)}"}
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"🗑️ Cleaned up temp file: {temp_file}")
            except Exception as e:
                print(f"⚠️ Failed to clean up {temp_file}: {e}")

def get_available_voices():
    """Get list of available predefined voices"""
    try:
        voices_dir = Path("/app/voices")
        if not voices_dir.exists():
            return {
                "success": True,
                "voices": [],
                "count": 0,
                "message": "Voices directory not found"
            }
        
        voice_files = []
        for file_path in voices_dir.glob("*.wav"):
            try:
                file_stat = file_path.stat()
                voice_files.append({
                    "filename": file_path.name,
                    "size_bytes": file_stat.st_size,
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 2)
                })
            except Exception as e:
                print(f"⚠️ Error reading voice file {file_path}: {e}")
        
        # Sort by filename
        voice_files.sort(key=lambda x: x["filename"])
        
        return {
            "success": True,
            "voices": voice_files,
            "count": len(voice_files)
        }
        
    except Exception as e:
        print(f"❌ Error getting voices: {str(e)}")
        return {"error": f"Failed to get voices: {str(e)}"}

def get_system_status():
    """Get system status and health information"""
    try:
        import torch
        
        status = {
            "success": True,
            "engine_initialized": engine is not None,
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "voices_directory_exists": os.path.exists("/app/voices")
        }
        
        if torch.cuda.is_available():
            status["gpu_info"] = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024 * 1024),
                "memory_cached_mb": torch.cuda.memory_reserved(0) / (1024 * 1024),
                "device_count": torch.cuda.device_count()
            }
        
        # Check if voices directory has files
        if os.path.exists("/app/voices"):
            voice_count = len(list(Path("/app/voices").glob("*.wav")))
            status["available_voices_count"] = voice_count
        
        return status
        
    except Exception as e:
        print(f"❌ Error getting status: {str(e)}")
        return {"error": f"Failed to get status: {str(e)}"}

def handler(job):
    """
    Main RunPod serverless handler function
    
    Supported request types:
    1. TTS Generation: {"type": "tts", "text": "...", ...}
    2. Get Voices: {"type": "get_voices"}  
    3. System Status: {"type": "status"}
    """
    
    try:
        # Ensure engine is initialized
        if not initialize_engine():
            return {"error": "Failed to initialize TTS engine"}
        
        # Parse job input
        job_input = job.get("input", {})
        request_type = job_input.get("type", "tts")  # Default to TTS
        
        print(f"📥 Received request type: {request_type}")
        
        # Route to appropriate handler
        if request_type == "tts":
            return process_tts_request(job_input)
        elif request_type == "get_voices":
            return get_available_voices()
        elif request_type == "status":
            return get_system_status()
        else:
            return {
                "error": f"Unknown request type: {request_type}",
                "supported_types": ["tts", "get_voices", "status"]
            }
    
    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return {"error": f"Handler error: {str(e)}"}

# Entry point for RunPod serverless
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless TTS Worker")
    print("=" * 50)
    
    # Test engine initialization
    if initialize_engine():
        print("✅ Engine pre-initialization successful")
        
        # Print some useful info
        if os.path.exists("/app/voices"):
            voice_count = len(list(Path("/app/voices").glob("*.wav")))
            print(f"📁 Found {voice_count} voice files in /app/voices")
        else:
            print("⚠️ /app/voices directory not found")
    else:
        print("❌ Engine pre-initialization failed")
        print("Worker will still start, but requests may fail")
    
    print("=" * 50)
    print("🎧 TTS Serverless Worker Ready!")
    
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})