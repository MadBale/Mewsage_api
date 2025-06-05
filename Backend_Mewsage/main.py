from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import librosa
import io
import os
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json


# Initialize app
app = FastAPI(title="Cat Sound Classifier API")

# Configure CORS for Flutter app compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model and label encoder (should be in the same directory)
MODEL_PATH = Path("catsound_class.keras")
LABEL_ENCODER_PATH = Path("label_encoder.json")

# Update your loading code in main.py
try:
    model = tf.keras.models.load_model('catsound_class.keras')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load label encoder with better error handling
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
        
    with open(LABEL_ENCODER_PATH, "r") as f:  # Explicitly open in read mode
        label_encoder_classes = json.load(f)
        
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)
    
except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
    print("Please ensure both 'catsound_class' and 'label_encoder.json' exist in the same directory")
    exit(1)  # Exit if files are missing

def process_audio_file(audio_data: bytes):
    """Process audio file from either upload or local path"""
    try:
        # Convert bytes to audio stream
        audio_stream = io.BytesIO(audio_data)
        audio, sr = librosa.load(audio_stream, sr=22050)
        
        # Feature extraction (same as your original)
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
        raise ValueError(f"Audio processing error: {str(e)}")

@app.post("/predict")
async def predict_from_upload(file: UploadFile = File(...)):
    """Endpoint for online prediction (file upload)"""
    try:
        audio_data = await file.read()
        features = process_audio_file(audio_data)
        
        # Make prediction
        proba = model.predict(features)[0]
        pred_class = np.argmax(proba)
        class_name = label_encoder.inverse_transform([pred_class])[0]
        
        return {
            "success": True,
            "prediction": class_name,
            "confidence": float(proba.max()),
            "probabilities": {label: float(prob) for label, prob in zip(label_encoder.classes_, proba)}
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict-local")
async def predict_from_local_path(file_path: str):
    """Endpoint for offline prediction (local file path)"""
    try:
        # Security check - validate file path
        if not os.path.exists(file_path):
            raise ValueError("File not found")
        
        # Read local file
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        features = process_audio_file(audio_data)
        
        # Make prediction
        proba = model.predict(features)[0]
        pred_class = np.argmax(proba)
        class_name = label_encoder.inverse_transform([pred_class])[0]
        
        return {
            "success": True,
            "prediction": class_name,
            "confidence": float(proba.max()),
            "probabilities": {label: float(prob) for label, prob in zip(label_encoder.classes_, proba)}
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# For testing offline functionality directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)