from __future__ import annotations

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceService:
    def __init__(self, model_name: str = "buffalo_l"):
        self.model = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        # The model is intentionally initialized once and reused for every request.
        self.model.prepare(ctx_id=0, det_size=(1280, 1280))

    @staticmethod
    def decode_image(contents: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image")
        return image

    def analyze(self, image: np.ndarray) -> list[dict]:
        faces = self.model.get(image)
        results = []
        for face in faces:
            results.append({
                "bbox": [int(value) for value in face.bbox],
                "detection_confidence": round(float(face.det_score), 4),
                "embedding": np.asarray(face.embedding, dtype=np.float32),
            })
        return results
