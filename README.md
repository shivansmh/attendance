# Attendance

Computer-vision-based attendance system.

## Current status

The original experiment in `Learning/test.py` detects and counts faces with InsightFace. A backend MVP is now in `backend/` with:

- one-face student enrollment;
- classroom face detection;
- embedding-based matching against enrolled students;
- an attendance response with present and unknown faces.

See [`backend/README.md`](backend/README.md) for setup and API examples.

## Recommended development order

1. Run the API on a small set of consented test images.
2. Build a labeled evaluation set and measure false matches/missed faces at different thresholds.
3. Add teacher review before any attendance record is finalized.
4. Move embeddings and attendance records to an authenticated database.
5. Add classroom/course/session models, audit logs, retention rules, and liveness checks.
6. Only then build the frontend around the stable API contract.
