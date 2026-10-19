# AI Planning Document — Intelligent Face Tracker

## 1. Problem
Count unique visitor identities in a video stream while logging each visible track's entry and exit with timestamps and cropped face images.

## 2. Users and use case
The prototype is intended for a controlled, consent-based demo using the hackathon's provided sample video. A future deployment would require a privacy review, signage/notice, retention limits, access controls, and a documented lawful basis.

## 3. Goals
- Detect faces using a YOLO face detector.
- Obtain face embeddings from InsightFace.
- Register new identities and reuse known identities where cosine similarity passes a configurable threshold.
- Track detections between detection cycles.
- Produce one entry and one exit event per track.
- Save image evidence locally and metadata in SQLite.
- Expose a unique identity count.

## 4. Out of scope
- Identifying people by their real-world names.
- Access-control or law-enforcement decisions.
- Cloud hosting or a frontend dashboard.
- Guaranteed identification under occlusion, poor lighting, or large pose changes.

## 5. Workflow
1. Load `config.json`.
2. Open a sample video, webcam, or RTSP stream.
3. Every `detection_skip_frames + 1` frames, run YOLO face detection.
4. Run InsightFace on the frame and associate its face embeddings with YOLO boxes by IoU.
5. Compare each embedding against stored embeddings using cosine similarity.
6. Reuse a matched `face_id`, or register a new UUID-based identity.
7. Associate current detections with existing tracks using greedy IoU matching.
8. Write an entry event for each new track and an exit event when a track has been missing too long or the run ends.
9. Save crops, write event metadata to SQLite, and append system events to `logs/events.log`.
10. Show the unique registered identity count.

## 6. Data model
- `visitors(face_id, embedding, first_seen)`
- `events(event_id, face_id, event_type, timestamp, image_path, track_id)`

## 7. Configuration
- `detection_skip_frames`: trade-off between compute and temporal coverage.
- `confidence_threshold`: YOLO detection confidence.
- `face_match_threshold`: identity matching threshold; must be validated on the sample data.
- `tracking_iou_threshold`: box overlap needed to match a track.
- `max_lost_detection_cycles`: missed detection cycles before exit.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Different YOLO and InsightFace boxes | Associate by IoU and skip uncertain associations |
| Same person registered twice | Tune matching threshold; evaluate lighting/pose cases |
| Different people merged | Increase threshold and inspect false-match cases |
| Missed short appearances | Reduce frame skip; test at target frame rate |
| Track switches during crossing | Use a stronger tracker and appearance features if needed |
| Database/file inconsistency | Log exceptions, use SQLite transactions, verify image paths |
| Sensitive biometric data exposed | Use consented demo footage, restrict access, do not push crops/DB to Git |
| Performance varies by hardware | Record measured FPS, CPU/GPU usage and resolution |

## 9. Validation plan
Run the same sample video with fixed settings. Count expected unique visitors manually for a short segment. Compare the app's unique count, inspect false matches/duplicate IDs, verify every entry has one exit for the run, verify image paths, and query database rows. Report results honestly and state the test segment and conditions.

## 10. Compute estimate plan
Do not invent benchmark numbers. Record:
- Machine CPU and RAM.
- GPU model and whether CUDA/ONNX GPU provider is active.
- Input resolution and FPS.
- Detection interval.
- Average processing FPS over a timed sample.
- Peak RAM/VRAM if available.
- CPU/GPU utilization while the stream runs.

## 11. AI-assisted development workflow
1. Convert the problem statement into requirements and acceptance tests.
2. Design modules and data schema before generating code.
3. Generate one module at a time.
4. Review every generated module and explain its logic.
5. Run syntax/import checks, then test each module.
6. Run a short sample clip and inspect the database, images, and logs.
7. Fix bugs based on observed outputs; do not claim untested functionality.
8. Document prompts, assumptions, test results, and known limitations for the interview.
