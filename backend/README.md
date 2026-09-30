# Attendance backend MVP

This service turns the current face-counting experiment into a small API:

- `POST /students/enroll`: enroll one student from a single-face image.
- `POST /detect`: count faces and return bounding boxes/confidence scores.
- `POST /attendance`: recognize enrolled students in a classroom image and return present/unknown faces.
- `GET /health`: service and registry status.

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first request downloads/loads the InsightFace `buffalo_l` model and may take a while.

## Try it

```bash
curl http://localhost:8000/health
curl -F student_id=cs001 -F name="Student One" -F image=@../Test\ Data/test.jpg http://localhost:8000/students/enroll
curl -F image=@../Test\ Data/test5.jpg http://localhost:8000/attendance
```

## Important MVP limitations

1. Face embeddings are currently stored in a local JSON file. Use a database with encryption and access control before real student data is used.
2. A classroom photo can produce false matches, missed faces, or duplicate detections. Attendance should have a teacher review step initially.
3. Add consent, retention/deletion rules, anti-spoofing/liveness checks, audit logs, and role-based authentication before deployment.
4. The match threshold (`FACE_MATCH_THRESHOLD`, default `0.45`) must be calibrated using your own classroom images. Do not treat the default as an accuracy guarantee.
5. This system identifies enrolled faces; face counting alone cannot mark who is present.
