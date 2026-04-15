"""
recognize.py
Run face recognition on:
  - Live webcam feed
  - A single image file
  - A video file

Usage:
    python recognize.py                        # webcam (default cam 0)
    python recognize.py --source 1             # webcam index 1
    python recognize.py --source image.jpg     # single image
    python recognize.py --source video.mp4     # video file
    python recognize.py --threshold 0.50       # override similarity threshold
"""

import argparse
import cv2
import numpy as np
import time

from config import (
    BOX_COLOR_KNOWN, BOX_COLOR_UNKNOWN,
    FONT_SCALE, THICKNESS, SIMILARITY_THRESHOLD,
)
from face_engine import FaceEngine
from gallery import FaceGallery


def draw_face(frame: np.ndarray, face, name: str, score: float):
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    known   = name != "unknown"
    color   = BOX_COLOR_KNOWN if known else BOX_COLOR_UNKNOWN
    label   = f"{name} ({score:.2f})" if known else f"Unknown ({score:.2f})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

    # Background for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), THICKNESS)


def process_frame(frame: np.ndarray, engine: FaceEngine, gallery: FaceGallery,
                  threshold: float) -> np.ndarray:
    faces = engine.get_faces(frame)

    for face in faces:
        emb = face.embedding
        if emb is None:
            continue
        name, score = gallery.search(emb, threshold=threshold)
        draw_face(frame, face, name, score)

    # HUD
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def run_webcam(cam_index: int, engine: FaceEngine, gallery: FaceGallery, threshold: float):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[Recognize] ERROR: Cannot open camera {cam_index}")
        return

    print(f"[Recognize] Camera {cam_index} opened. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, engine, gallery, threshold)

        # FPS overlay
        now   = time.time()
        fps   = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str, engine: FaceEngine, gallery: FaceGallery, threshold: float):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[Recognize] ERROR: Cannot read '{path}'")
        return
    result = process_frame(frame, engine, gallery, threshold)
    cv2.imshow("Face Recognition", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path: str, engine: FaceEngine, gallery: FaceGallery, threshold: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Recognize] ERROR: Cannot open '{path}'")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, engine, gallery, threshold)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--source",    default="0",  help="Camera index, image, or video path")
    parser.add_argument("--threshold", type=float,   default=SIMILARITY_THRESHOLD,
                        help="Cosine similarity threshold (default: from config)")
    args = parser.parse_args()

    gallery = FaceGallery()
    if gallery.index.ntotal == 0:
        print("[Recognize] WARNING: Gallery is empty. Enroll people first with enroll.py.")

    engine = FaceEngine()

    src = args.source
    if src.isdigit():
        run_webcam(int(src), engine, gallery, args.threshold)
    elif src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_image(src, engine, gallery, args.threshold)
    else:
        run_video(src, engine, gallery, args.threshold)


if __name__ == "__main__":
    main()
