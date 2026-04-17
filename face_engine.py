"""
face_engine.py
Face detection and embedding using simpler OpenCV-based approach.
Works reliably on any platform without model downloads.
"""

import cv2
import numpy as np
from config import DET_SIZE


class FaceEngine:
    _instance = None
    _face_cascade = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        print("[FaceEngine] Loading OpenCV face detection...")
        
        # Use Haar Cascade - no download needed, comes with OpenCV
        cascade_path = cv2.data.haarcascades
        
        # Try multiple cascade files
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml', 
            'haarcascade_frontalface_alt2.xml',
        ]
        
        for casc in cascade_files:
            try:
                self._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + casc
                )
                if not self._face_cascade.empty():
                    print(f"[FaceEngine] Loaded: {casc}")
                    break
            except:
                continue
        
        if self._face_cascade is None or self._face_cascade.empty():
            raise RuntimeError("Could not load any face cascade")
        
        self._initialized = True
        print("[FaceEngine] Ready.")

    def get_faces(self, img: np.ndarray) -> list:
        """Detect all faces in image."""
        if img is None or img.size == 0:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to Face-like objects
        results = []
        for (x, y, w, h) in faces:
            face = type('Face', (), {})()
            face.bbox = np.array([x, y, x + w, y + h])
            face.embedding = self._get_simple_embedding(img, face.bbox)
            results.append(face)
        
        # Sort by size
        results.sort(key=lambda f: _face_area(f), reverse=True)
        return results

    def get_largest_face(self, img: np.ndarray):
        """Return largest face."""
        faces = self.get_faces(img)
        return faces[0] if faces else None

    def embed_image(self, img: np.ndarray) -> np.ndarray | None:
        """Get embedding for largest face."""
        face = self.get_largest_face(img)
        if face is None:
            return None
        return face.embedding

    def embed_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """Get embedding for face crop."""
        if crop is None or crop.size == 0:
            return None
        
        # Resize to consistent size
        crop = cv2.resize(crop, (128, 128))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Simple histogram-based embedding
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        
        # Pad to 512 dim
        embedding = np.zeros(512, dtype=np.float32)
        embedding[:len(hist)] = hist
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def _get_simple_embedding(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Create simple embedding from face region."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract and resize face
        face_img = img[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (128, 128))
        
        # Simple feature: histogram + basic stats
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
        hist = hist.flatten()
        
        # Basic statistics
        stats = [
            gray.mean(), gray.std(),
            gray.min(), gray.max(),
        ]
        stats = np.array(stats, dtype=np.float32)
        
        # Combine
        embedding = np.concatenate([hist, stats])
        
        # Pad to 512
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    return float(np.dot(a, b))


# Keep InsightFace for comparison (optional)
try:
    from insightface.app import FaceAnalysis as _InsightFace
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
