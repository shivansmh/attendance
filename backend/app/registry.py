"""Small JSON-backed registry for the first local MVP.

Replace this with Postgres/Supabase and encrypted object storage before production.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


class FaceRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.path.exists():
            self._write({"students": {}})

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        temporary.replace(self.path)

    def upsert(self, student_id: str, name: str, embedding: np.ndarray) -> None:
        normalized = embedding / (np.linalg.norm(embedding) or 1.0)
        with self._lock:
            data = self._read()
            data["students"][student_id] = {
                "student_id": student_id,
                "name": name,
                "embedding": normalized.astype(float).tolist(),
            }
            self._write(data)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read()["students"].values())

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[dict[str, Any] | None, float]:
        normalized = embedding / (np.linalg.norm(embedding) or 1.0)
        best_student = None
        best_score = -1.0
        for student in self.all():
            enrolled = np.asarray(student["embedding"], dtype=np.float32)
            score = float(np.dot(normalized, enrolled))
            if score > best_score:
                best_student, best_score = student, score
        if best_student is None or best_score < threshold:
            return None, best_score
        return best_student, best_score
