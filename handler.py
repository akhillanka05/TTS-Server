import runpod
import torch
import torchaudio
import base64
import io
import logging
from pathlib import Path

# Import your existing engine module
from engine import load_model, synthesize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(job):
    """
    RunPod serverless handler for Chatterbox TTS
    
    Expected input:
    {
        "text": "Text to synthesize",
        "audio_prompt_path": "optional/path/to/reference/audio.wav",
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "seed": 0
    }
    
    Returns:
    {
        "audio_base64": "base64 encoded audio data",
        "sample_rate": 24000
    }
    """
    try:
        # Get job inputs
        job_input = job['input']
        
        # Required parameter
        text = job_input.get('text', '')
        if not text:
            return {"error": "Text is required"}
        
        # Optional parameters with defaults
        audio_prompt_path = job_input.get('audio_prompt_path', None)
        temperature = job_input.get('temperature', 0.8)
        exaggeration = job_input.get('exaggeration', 0.5)
        cfg_weight = job_input.get('cfg_weight', 0.5)
        seed = job_input.get('seed', 0)
        
        logger.info(f"Generating TTS for text: {text[:50]}...")
        
        # Generate audio using your existing synthesize function
        wav_tensor, sample_rate = synthesize(
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed
        )
        
        if wav_tensor is None:
            return {"error": "Failed to generate audio"}
        
        # Convert tensor to base64 encoded audio
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor.unsqueeze(0), sample_rate, format="wav")
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}

def init():
    """
    Initialize the model when the container starts.
    This runs once and keeps the model loaded in memory.
    """
    logger.info("Initializing Chatterbox TTS model...")
    
    try:
        # Load the model using your existing function
        success = load_model()
        if success:
            logger.info("Model loaded successfully!")
        else:
            logger.error("Failed to load model")
            raise Exception("Model loading failed")
            
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise e

if __name__ == "__main__":
    # Initialize model once
    init()
    
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})