import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import numpy as np
import librosa
import io
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Cat Sound Classifier API",
    description="API for classifying cat sounds using CNN+LSTM model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Security - API Key (set via environment variable in Render)
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

# Configure CORS - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your Flutter app's domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = Path("best_model.keras")
LABEL_ENCODER_PATH = Path("label_encoder.json")
MAX_FILE_SIZE_MB = 5  # 5MB max file size
ALLOWED_CONTENT_TYPES = ["audio/wav", "audio/mpeg", "audio/x-wav"]

# Load model and label encoder at startup
@app.on_event("startup")
async def load_model_and_encoder():
    try:
        app.state.model = tf.keras.models.load_model(MODEL_PATH)
        app.state.model.compile(optimizer='adam', 
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])
        
        with open(LABEL_ENCODER_PATH) as f:
            label_encoder_classes = json.load(f)
        app.state.label_encoder = LabelEncoder()
        app.state.label_encoder.classes_ = np.array(label_encoder_classes)
        
        logger.info("Model and label encoder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model or label encoder: {str(e)}")
        raise

def validate_audio_file(file: UploadFile):
    """Validate the uploaded audio file"""
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset pointer
    
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB}MB"
        )
    
    # Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type. Allowed: {ALLOWED_CONTENT_TYPES}"
        )

def process_audio_file(audio_data: bytes) -> np.ndarray:
    """Process audio file into model-ready features"""
    try:
        # Convert bytes to audio stream
        audio_stream = io.BytesIO(audio_data)
        audio, sr = librosa.load(audio_stream, sr=22050)
        
        # Feature extraction
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad/truncate to 105 time steps
        max_pad_len = 105
        if mel_spec_db.shape[1] > max_pad_len:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
        
        # Normalize
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
        return mel_spec_db.T[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file. Please provide a valid WAV or MP3 file."
        )

@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint for Render monitoring"""
    return Response(status_code=200)

@app.post("/predict")
async def predict_from_upload(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """Endpoint for cat sound classification from audio upload"""
    try:
        # Validate file
        validate_audio_file(file)
        
        # Read and process file
        audio_data = await file.read()
        features = process_audio_file(audio_data)
        
        # Make prediction
        proba = app.state.model.predict(features)[0]
        pred_class = np.argmax(proba)
        class_name = app.state.label_encoder.inverse_transform([pred_class])[0]
        
        # Format probabilities
        probabilities = {
            label: float(prob) 
            for label, prob in zip(app.state.label_encoder.classes_, proba)
        }
        
        logger.info(f"Prediction successful: {class_name} (confidence: {proba.max():.2f})")
        
        return {
            "success": True,
            "prediction": class_name,
            "confidence": float(proba.max()),
            "probabilities": probabilities
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again."
        )

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return Response(
        content=json.dumps({
            "success": False,
            "error": "Internal server error"
        }),
        status_code=500,
        media_type="application/json"
    )

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
