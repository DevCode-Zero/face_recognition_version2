"""
enroll.py
CLI tool to enroll persons into the face gallery.
Uses Supabase for storage.
"""

import argparse
import cv2

from config import AUGMENT_COUNT
from face_engine import FaceEngine
from augment import augment_face_crop
from gallery import FaceGallery


def enroll_person(name: str, image_paths: list[str], gallery: FaceGallery, engine: FaceEngine):
    all_embeddings = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Enroll] WARNING: Cannot read '{img_path}', skipping.")
            continue

        face = engine.get_largest_face(img)
        if face is None:
            print(f"[Enroll] WARNING: No face detected in '{img_path}', skipping.")
            continue

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad = int((x2 - x1) * 0.15)
        h, w = img.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]

        variants = augment_face_crop(crop, n=AUGMENT_COUNT)
        print(f"[Enroll] '{img_path}' → {len(variants)} augmented variants")

        embedded = 0
        for variant in variants:
            emb = engine.embed_crop(variant)
            if emb is not None:
                all_embeddings.append(emb)
                embedded += 1

        orig_emb = face.embedding
        if orig_emb is not None:
            all_embeddings.append(orig_emb)
            embedded += 1

        print(f"[Enroll] Successfully embedded {embedded} variants from '{img_path}'")

    if not all_embeddings:
        print(f"[Enroll] ERROR: No embeddings generated for '{name}'. Enrollment failed.")
        return False

    gallery.add_embeddings(name, all_embeddings)
    print(f"[Enroll] ✓ '{name}' enrolled with {len(all_embeddings)} total embeddings.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Face Enrollment CLI (Supabase)")
    parser.add_argument("--name",   type=str, help="Person's full name")
    parser.add_argument("--images", nargs="+", help="Path(s) to enrollment photo(s)")
    parser.add_argument("--list",   action="store_true", help="List all enrolled people")
    parser.add_argument("--remove", type=str, help="Remove a person from gallery")
    args = parser.parse_args()

    gallery = FaceGallery()

    if args.list:
        names = gallery.enrolled_names()
        counts = gallery.embedding_count_per_person()
        print(f"\n{'─'*40}")
        print(f"{'Name':<25} {'Embeddings':>10}")
        print(f"{'─'*40}")
        for n in names:
            print(f"{n:<25} {counts[n]:>10}")
        print(f"{'─'*40}")
        print(f"Total people: {len(names)}, Total embeddings: {sum(counts.values())}")
        return

    if args.remove:
        gallery.remove_person(args.remove)
        return

    if not args.name or not args.images:
        parser.print_help()
        return

    engine = FaceEngine()
    enroll_person(args.name, args.images, gallery, engine)


if __name__ == "__main__":
    main()
