"""
gallery.py
Face gallery - supports both local JSON storage and Supabase + pgvector.
"""

import os
import json
import numpy as np
from config import SUPABASE_URL, SUPABASE_KEY, TABLE_PREFIX, EMBEDDING_DIM, SIMILARITY_THRESHOLD

try:
    from supabase import create_client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False


class FaceGallery:
    def __init__(self):
        self.dim = EMBEDDING_DIM
        self.use_local = not (SUPABASE_URL and SUPABASE_KEY and HAS_SUPABASE)
        
        if self.use_local:
            print("[Gallery] Using LOCAL storage (no Supabase credentials)")
            self._local_data = {}
            self._load_local()
        else:
            self.table = f"{TABLE_PREFIX}embeddings"
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print(f"[Gallery] Connected to Supabase (table: {self.table})")
            self._verify_connection()

    def _load_local(self):
        self._local_file = "gallery_data.json"
        if os.path.exists(self._local_file):
            with open(self._local_file, "r") as f:
                self._local_data = json.load(f)
            print(f"[Gallery] Loaded {sum(len(v) for v in self._local_data.values())} embeddings from local storage")

    def _save_local(self):
        with open(self._local_file, "w") as f:
            json.dump(self._local_data, f)

    def _verify_connection(self):
        try:
            self.client.table(self.table).select("id").limit(1).execute()
            print(f"[Gallery] Database connection verified")
        except Exception as e:
            raise RuntimeError(f"[Gallery] Database connection failed: {e}")

    def _normalize_embedding(self, embedding: np.ndarray) -> list:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.tolist()
        return (embedding / norm).tolist()

    def add_embeddings(self, name: str, embeddings: list):
        normalized = [self._normalize_embedding(emb) for emb in embeddings]
        
        if self.use_local:
            if name not in self._local_data:
                self._local_data[name] = []
            self._local_data[name].extend(normalized)
            self._save_local()
            print(f"[Gallery] Added {len(embeddings)} embeddings for '{name}'")
        else:
            records = [{"name": name, "embedding": emb} for emb in normalized]
            self.client.table(self.table).insert(records).execute()
            print(f"[Gallery] Added {len(embeddings)} embeddings for '{name}'")

    def remove_person(self, name: str):
        if self.use_local:
            if name in self._local_data:
                del self._local_data[name]
                self._save_local()
            print(f"[Gallery] Removed '{name}'")
        else:
            self.client.table(self.table).delete().eq("name", name).execute()
            print(f"[Gallery] Removed '{name}'")

    def search(self, embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD):
        normalized = self._normalize_embedding(embedding)
        
        if self.use_local:
            return self._local_search(normalized, threshold)
        
        try:
            result = self.client.rpc(
                f"{TABLE_PREFIX}match_embeddings",
                {
                    "query_embedding": normalized,
                    "match_threshold": threshold,
                    "match_count": 5
                }
            ).execute()
            
            if not result.data:
                return self._fallback_search(normalized, threshold)
            
            best = result.data[0]
            similarity = best.get("similarity", 0)
            
            if similarity >= threshold:
                return best["name"], similarity
            return "unknown", similarity
        except Exception as e:
            print(f"[Gallery] RPC search failed: {e}")
            return self._fallback_search(normalized, threshold)

    def _parse_embedding(self, emb_data):
        if isinstance(emb_data, list):
            return np.array(emb_data, dtype=np.float32)
        if isinstance(emb_data, str):
            import ast
            return np.array(ast.literal_eval(emb_data), dtype=np.float32)
        return np.array(emb_data)

    def _local_search(self, normalized: np.ndarray, threshold: float):
        if not self._local_data:
            return "unknown", 0.0
        
        person_scores = {}
        for name, embeddings in self._local_data.items():
            for emb in embeddings:
                emb_arr = self._parse_embedding(emb)
                score = float(np.dot(normalized, emb_arr))
                person_scores[name] = max(person_scores.get(name, -1), score)
        
        if not person_scores:
            return "unknown", 0.0
        
        best_name = max(person_scores, key=person_scores.get)
        best_score = person_scores[best_name]
        
        if best_score >= threshold:
            return best_name, best_score
        return "unknown", best_score

    def _fallback_search(self, normalized: np.ndarray, threshold: float):
        result = self.client.table(self.table).select("name, embedding").execute()
        if not result.data:
            return "unknown", 0.0
        
        person_scores = {}
        
        for row in result.data:
            emb = self._parse_embedding(row["embedding"])
            score = float(np.dot(normalized, emb))
            person_scores[row["name"]] = max(person_scores.get(row["name"], -1), score)
        
        if not person_scores:
            return "unknown", 0.0
        
        best_name = max(person_scores, key=person_scores.get)
        best_score = person_scores[best_name]
        
        if best_score >= threshold:
            return best_name, best_score
        return "unknown", best_score

    def save(self):
        if self.use_local:
            self._save_local()
            print(f"[Gallery] Data persisted locally")
        else:
            print(f"[Gallery] Data persisted in Supabase")

    def enrolled_names(self) -> list:
        if self.use_local:
            return sorted(self._local_data.keys())
        result = self.client.table(self.table).select("name").execute()
        names = list(set(row["name"] for row in result.data))
        return sorted(names)

    def embedding_count_per_person(self) -> dict:
        if self.use_local:
            return {name: len(embs) for name, embs in self._local_data.items()}
        result = self.client.table(self.table).select("name").execute()
        from collections import Counter
        names = [row["name"] for row in result.data]
        return dict(Counter(names))

    def count(self) -> int:
        if self.use_local:
            return sum(len(v) for v in self._local_data.values())
        result = self.client.table(self.table).select("id", count="exact").execute()
        return result.count