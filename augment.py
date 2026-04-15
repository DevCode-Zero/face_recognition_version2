"""
augment.py
Generates synthetic face variants from a single enrollment image.
Goal: turn 1-2 real photos into 20-40 embeddings that cover realistic
variation (lighting, slight angle, quality degradation).
"""

import cv2
import numpy as np
import random
from config import AUGMENT_COUNT


def _clahe(img: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def _brightness_contrast(img, alpha_range=(0.7, 1.4), beta_range=(-30, 30)):
    alpha = random.uniform(*alpha_range)
    beta  = random.randint(*beta_range)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _gaussian_blur(img, max_k=3):
    k = random.choice([1, 3]) if max_k >= 3 else 1
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def _horizontal_flip(img):
    return cv2.flip(img, 1)


def _rotate(img, max_deg=12):
    deg = random.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _perspective_warp(img, max_shift=0.05):
    h, w = img.shape[:2]
    margin = int(min(h, w) * max_shift)
    pts1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts2 = pts1 + np.array([
        [random.randint(-margin, margin), random.randint(-margin, margin)]
        for _ in range(4)
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _jpeg_compression(img, quality_range=(50, 95)):
    q = random.randint(*quality_range)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _add_noise(img, sigma_range=(2, 15)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ─── Augmentation pipeline ────────────────────────────────────────────────

AUGMENTERS = [
    _brightness_contrast,
    _gaussian_blur,
    _horizontal_flip,
    _rotate,
    _perspective_warp,
    _jpeg_compression,
    _add_noise,
    _clahe,
]


def augment_face_crop(face_crop: np.ndarray, n: int = AUGMENT_COUNT) -> list:
    """
    Given a tightly-cropped aligned face (BGR numpy array),
    return a list of `n` augmented variants including the original.
    """
    variants = [face_crop.copy()]  # always include original

    for _ in range(n - 1):
        aug = face_crop.copy()
        # apply 1-3 random augmenters
        chosen = random.sample(AUGMENTERS, k=random.randint(1, 3))
        for fn in chosen:
            try:
                aug = fn(aug)
            except Exception:
                pass  # skip failed augment, keep previous state
        variants.append(aug)

    return variants
