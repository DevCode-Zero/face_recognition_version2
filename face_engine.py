"""
face_engine.py
Wraps InsightFace for face detection and embedding.
"""

import os
import cv2
import numpy as np
import urllib.request
import zipfile
import time
import insightface
from insightface.app import FaceAnalysis
from config import INSIGHTFACE_MODEL, DET_SIZE, CTX_ID


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def _ensure_models():
    """Download models if not present."""
    # Try multiple base directories
    base_dirs = [
        os.path.expanduser("~/.insightface"),
        "/tmp/.insightface", 
        os.path.join(os.getcwd(), ".insightface"),
    ]
    
    model_dir = None
    
    # Check existing models
    for base_dir in base_dirs:
        test_dir = os.path.join(base_dir, "models", "buffalo_l")
        if os.path.exists(test_dir):
            onnx_files = [f for f in os.listdir(test_dir) if f.endswith('.onnx')]
            if len(onnx_files) >= 3:
                model_dir = test_dir
                os.environ["INSIGHTFACE_MODEL_ROOT"] = os.path.join(base_dir, "models")
                print(f"[FaceEngine] Found models at: {model_dir}")
                print(f"[FaceEngine] Model files: {onnx_files}")
                return
    
    # Need to download
    for base_dir in base_dirs:
        try:
            model_dir = os.path.join(base_dir, "models", "buffalo_l")
            os.makedirs(model_dir, exist_ok=True)
            os.environ["INSIGHTFACE_MODEL_ROOT"] = os.path.join(base_dir, "models")
            
            print(f"[FaceEngine] Downloading models to {model_dir}...")
            
            zip_path = os.path.join(base_dir, "buffalo_l.zip")
            
            # Download with timeout and headers
            request = urllib.request.Request(
                MODEL_URL,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                total_size = int(response.headers.get('content-length', 0))
                print(f"[FaceEngine] Total size: {total_size / 1024 / 1024:.1f} MB")
                
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192*32)
                        if not chunk:
                            break
                        f.write(chunk)
            
            # Extract
            print(f"[FaceEngine] Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(os.path.join(base_dir, "models"))
            os.remove(zip_path)
            
            print(f"[FaceEngine] Download complete!")
            return
            
        except Exception as e:
            print(f"[FaceEngine] Failed with {base_dir}: {e}")
            continue
    
    raise RuntimeError("Could not download models from any location")


class FaceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        _ensure_models()
        
        # Set model root explicitly
        model_root = os.environ.get("INSIGHTFACE_MODEL_ROOT", "")
        print(f"[FaceEngine] Model root: {model_root}")
        
        print(f"[FaceEngine] Loading InsightFace model '{INSIGHTFACE_MODEL}'...")
        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
        self._initialized = True
        print("[FaceEngine] Ready.")

    # --- Public API ---

    def get_faces(self, img: np.ndarray) -> list:
        """Detect all faces in image."""
        faces = self.app.get(img)
        faces = sorted(faces, key=lambda f: _face_area(f), reverse=True)
        return faces

    def get_largest_face(self, img: np.ndarray):
        """Return the single largest face."""
        faces = self.get_faces(img)
        return faces[0] if faces else None

    def embed_image(self, img: np.ndarray) -> np.ndarray | None:
        """Detect and embed largest face."""
        face = self.get_largest_face(img)
        if face is None:
            return None
        return face.embedding

    def embed_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """Embed a face crop."""
        return self.embed_image(crop)


# --- Helper ---

def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between embeddings."""
    return float(np.dot(a, b))
