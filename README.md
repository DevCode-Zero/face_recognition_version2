# Face Recognition System
## Few-shot | ArcFace embeddings | FAISS search | CPU-only

---

## Setup

```bash
# 1. Create & activate venv
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. First run downloads InsightFace buffalo_l model (~500MB, auto)
```

---

## Enroll People

```bash
# Enroll with 1 photo
python enroll.py --name "Shivam Kumar" --images shivam.jpg

# Enroll with 2 photos (recommended)
python enroll.py --name "John Doe" --images john1.jpg john2.jpg

# List all enrolled people
python enroll.py --list

# Remove a person
python enroll.py --remove "John Doe"
```

---

## Run Recognition

```bash
# Webcam (default)
python recognize.py

# Specific camera
python recognize.py --source 1

# On an image
python recognize.py --source test.jpg

# On a video file
python recognize.py --source footage.mp4

# Override threshold (0.0–1.0, higher = stricter)
python recognize.py --threshold 0.50
```

---

## Threshold Tuning

The default threshold is **0.45** (in `config.py`).

| Threshold | Effect |
|-----------|--------|
| 0.35 | Looser — fewer "Unknown", more false accepts |
| 0.45 | Default — balanced |
| 0.55 | Stricter — fewer false accepts, more "Unknown" |

**To tune**: enroll your people → test in your actual environment → adjust
`SIMILARITY_THRESHOLD` in `config.py` until false positives disappear.

---

## Architecture

```
Camera Frame
    ↓
RetinaFace Detection       (finds face bbox + 5-point landmarks)
    ↓
Alignment + Preprocessing  (handled by InsightFace internally)
    ↓
ArcFace Embedding          (512-d L2-normalized vector)
    ↓
FAISS Inner Product Search (cosine similarity against gallery)
    ↓
Max-score aggregation      (best match across all enrolled embeddings)
    ↓
Threshold check → Name / Unknown
```

---

## Why This Works With 1-2 Photos

- **buffalo_l** ArcFace is pretrained on millions of faces — it already understands faces deeply
- **Augmentation at enrollment**: 1 photo → 20 synthetic variants covering lighting, angle, blur
- **Max-score matching**: query matches the best embedding in the gallery, not the average
- **FAISS IndexFlatIP**: exact nearest neighbor, no approximation errors for <50 people

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings (paths, thresholds, model) |
| `face_engine.py` | InsightFace wrapper (detect + embed) |
| `augment.py` | Synthetic augmentation at enrollment |
| `gallery.py` | FAISS index + metadata management |
| `enroll.py` | CLI enrollment tool |
| `recognize.py` | CLI recognition (webcam / image / video) |
