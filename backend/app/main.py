from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .face_service import FaceService
from .registry import FaceRegistry

APP_DIR = Path(__file__).resolve().parent
registry = FaceRegistry(os.getenv("FACE_REGISTRY_PATH", APP_DIR / "data" / "students.json"))
face_service: FaceService | None = None
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))

app = FastAPI(title="Attendance API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    enrolled_students: int


class DetectionResponse(BaseModel):
    face_count: int
    faces: list[dict]


class AttendanceRecord(BaseModel):
    student_id: str | None
    name: str | None
    match_score: float | None
    bbox: list[int] = Field(min_length=4, max_length=4)
    detection_confidence: float


class AttendanceResponse(BaseModel):
    captured_at: str
    face_count: int
    present_count: int
    unknown_count: int
    attendance: list[AttendanceRecord]


def get_face_service() -> FaceService:
    global face_service
    if face_service is None:
        face_service = FaceService(os.getenv("INSIGHTFACE_MODEL", "buffalo_l"))
    return face_service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", enrolled_students=len(registry.all()))


@app.post("/students/enroll")
async def enroll_student(
    student_id: Annotated[str, Form(min_length=1)],
    name: Annotated[str, Form(min_length=1)],
    image: Annotated[UploadFile, File(...)],
):
    contents = await image.read()
    try:
        faces = get_face_service().analyze(get_face_service().decode_image(contents))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if len(faces) != 1:
        raise HTTPException(status_code=400, detail=f"Enrollment requires exactly one face; found {len(faces)}")
    registry.upsert(student_id.strip(), name.strip(), faces[0]["embedding"])
    return {"student_id": student_id.strip(), "name": name.strip(), "status": "enrolled"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(image: UploadFile = File(...)) -> DetectionResponse:
    contents = await image.read()
    try:
        faces = get_face_service().analyze(get_face_service().decode_image(contents))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return DetectionResponse(
        face_count=len(faces),
        faces=[{key: value for key, value in face.items() if key != "embedding"} for face in faces],
    )


@app.post("/attendance", response_model=AttendanceResponse)
async def create_attendance(image: UploadFile = File(...)) -> AttendanceResponse:
    contents = await image.read()
    try:
        faces = get_face_service().analyze(get_face_service().decode_image(contents))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    records = []
    present_ids = set()
    for face in faces:
        student, score = registry.match(face["embedding"], MATCH_THRESHOLD)
        student_id = student["student_id"] if student else None
        if student_id:
            present_ids.add(student_id)
        records.append(AttendanceRecord(
            student_id=student_id,
            name=student["name"] if student else None,
            match_score=round(score, 4) if student else None,
            bbox=face["bbox"],
            detection_confidence=face["detection_confidence"],
        ))

    return AttendanceResponse(
        captured_at=datetime.now(timezone.utc).isoformat(),
        face_count=len(faces),
        present_count=len(present_ids),
        unknown_count=sum(record.student_id is None for record in records),
        attendance=records,
    )
