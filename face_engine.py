"""
face_engine.py
Wraps InsightFace for:
  - Face detection (RetinaFace)
  - Landmark-based alignment
  - ArcFace embedding extraction
"""

import os
import cv2
import numpy as np
import urllib.request
import zipfile
import insightface
from insightface.app import FaceAnalysis
from config import INSIGHTFACE_MODEL, DET_SIZE, CTX_ID


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_DIR = os.path.expanduser("~/.insightface/models/buffalo_l")


def _ensure_models():
    """Download models if not present."""
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        return  # Already downloaded
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, "..", "buffalo_l.zip")
    
    print(f"[FaceEngine] Downloading models...")
    try:
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(os.path.join(MODEL_DIR, ".."))
        os.remove(zip_path)
        print(f"[FaceEngine] Models downloaded.")
    except Exception as e:
        print(f"[FaceEngine] Download failed: {e}")
        raise


class FaceEngine:
    _instance = None  # singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Download models if needed
        _ensure_models()
        
        print(f"[FaceEngine] Loading InsightFace model '{INSIGHTFACE_MODEL}' on CPU...")
        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
        self._initialized = True
        print("[FaceEngine] Ready.")

    # ─── Public API ───────────────────────────────────────────────────────

    def get_faces(self, img: np.ndarray) -> list:
        """
        Detect all faces in `img` (BGR).
        Returns list of InsightFace Face objects, sorted largest → smallest.
        Each face has: .bbox, .landmark_2d_106, .embedding (512-d L2-normed).
        """
        faces = self.app.get(img)
        faces = sorted(faces, key=lambda f: _face_area(f), reverse=True)
        return faces

    def get_largest_face(self, img: np.ndarray):
        """Return the single largest face, or None if none detected."""
        faces = self.get_faces(img)
        return faces[0] if faces else None

    def embed_image(self, img: np.ndarray) -> np.ndarray | None:
        """
        Detect + embed the largest face in a full image.
        Returns 512-d normalized embedding, or None if no face found.
        """
        face = self.get_largest_face(img)
        if face is None:
            return None
        return face.embedding  # already L2-normalized by InsightFace

    def embed_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Embed a face crop (aligned or not).
        InsightFace will detect + align internally.
        Returns embedding or None.
        """
        return self.embed_image(crop)


# ─── Helper ───────────────────────────────────────────────────────────────

def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(a, b))  # already normalized, dot = cosine
