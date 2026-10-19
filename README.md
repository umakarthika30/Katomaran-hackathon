# Katomaran Intelligent Face Tracker

A Python prototype for face detection, face re-identification, track lifecycle events, timestamped face crops, SQLite storage, and unique-visitor counting.

> **Important:** This is a hackathon prototype, not a production biometric-security system. Use it only with appropriate notice/consent and follow applicable privacy and data-protection rules. Do not use it to make decisions about a person's rights, access, employment, or safety.

## 1. Problem statement in simple words

The application reads a video or camera stream, detects faces, assigns a persistent `face_id` to a previously unseen face, recognizes that identity when it appears again, and follows each visible appearance as a track. Each track produces one `entry` event and one `exit` event, including a timestamp and a cropped face image. Visitor metadata and events are saved in SQLite; operational details are written to `logs/events.log`.

The **unique visitor count** is the number of registered identities in the `visitors` database table. Seeing an already registered identity again must not increase that count.

## 2. Features

- YOLO-based face-box detection (face-specific YOLO weights required).
- InsightFace embeddings for face re-identification.
- Greedy IoU tracking across detection cycles.
- Automatic first-seen registration with IDs such as `visitor_a1b2c3d4`.
- SQLite database for visitors and entry/exit events.
- Cropped images saved under `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`.
- `logs/events.log` records registration, recognition, entry, exit, errors, and summary.
- `config.json` controls detection skipping and matching/tracking thresholds.
- Video file, webcam index, or RTSP stream as input.
- Live display with track IDs and unique visitor count.

## 3. Architecture

```mermaid
flowchart TD
    A[Video file / Webcam / RTSP] --> B[YOLO face detector]
    B --> C[Face bounding boxes]
    C --> D[InsightFace embedding extraction]
    D --> E[SQLite identity matching]
    E -->|New identity| F[Register face_id + embedding]
    E -->|Known identity| G[Reuse existing face_id]
    F --> H[IoU tracking]
    G --> H
    H --> I[Entry / Exit event manager]
    I --> J[Timestamped face crops]
    I --> K[(SQLite events table)]
    F --> L[(SQLite visitors table)]
    I --> M[logs/events.log]
    K --> N[Unique visitor count / on-screen overlay]
    L --> N
```

## 4. Project structure

```text
katomaran_face_tracker/
├── config.json
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── yolov8n-face.pt       # Add compatible face-specific YOLO weights here
├── data/
│   └── visitors.db           # Created automatically
├── logs/
│   ├── events.log            # Created automatically
│   ├── entries/YYYY-MM-DD/
│   └── exits/YYYY-MM-DD/
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── face_pipeline.py
│   ├── logger_setup.py
│   ├── tracker.py
│   └── main.py
└── docs/
    ├── AI_PLANNING.md
    ├── INTERVIEW_QA.md
    └── WORKFLOW.md
```

## 5. Requirements

- Windows, macOS, or Linux.
- Python 3.10 or 3.11 recommended.
- Git and VS Code.
- Internet access for initial Python package/model setup.
- A compatible face-specific YOLO `.pt` weights file at `models/yolov8n-face.pt`.
- The first InsightFace model initialization may download its model pack; make sure your environment permits this.

The standard `yolov8n.pt` COCO model is **not a face-specific model** and is not a drop-in replacement. Use face-detection weights whose license and source you have checked.

## 6. Setup in VS Code

Open the project folder in VS Code, then open Terminal → New Terminal.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal, if allowed by your organization's policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is not available, install Python 3.10/3.11 and reopen VS Code.

### Add the model and video

1. Put your compatible face-specific YOLO weights at `models/yolov8n-face.pt`.
2. Download the provided sample video using the link in the official problem statement and save it as `sample.mp4` in the project root.
3. Confirm `config.json` contains the correct paths.

### Run

```powershell
python -m src.main
```

Press **Q** or **Esc** to stop. The program also closes when the video reaches its end.

## 7. Input configuration

Example `config.json`:

```json
{
  "video_source": "sample.mp4",
  "yolo_model_path": "models/yolov8n-face.pt",
  "insightface_model": "buffalo_l",
  "detection_skip_frames": 2,
  "confidence_threshold": 0.45,
  "face_match_threshold": 0.45,
  "tracking_iou_threshold": 0.30,
  "max_lost_detection_cycles": 8,
  "database_path": "data/visitors.db",
  "logs_root": "logs",
  "window_name": "Katomaran Intelligent Face Tracker"
}
```

- `detection_skip_frames: 2` means run detection once, then skip two frames; detection runs every three frames.
- `face_match_threshold` is a starting value, not a universal correct threshold. Evaluate it on the provided sample video and tune it to reduce false matches and duplicate identities.
- `tracking_iou_threshold` controls how much bounding boxes must overlap to be considered the same track.
- `max_lost_detection_cycles` controls how many consecutive detection cycles may miss a track before an exit event is emitted.
- For a webcam, set `"video_source": "0"`.
- For an RTSP camera, set `"video_source": "rtsp://username:password@host:port/path"` and keep credentials out of Git. Prefer environment variables or a local untracked config for secrets.

## 8. Database

SQLite creates these tables automatically:

### `visitors`
- `face_id`: unique registered identity.
- `embedding`: normalized facial embedding stored as a binary blob.
- `first_seen`: first registration timestamp.

### `events`
- `event_id`: unique event row.
- `face_id`: registered identity.
- `event_type`: `entry` or `exit`.
- `timestamp`: event timestamp with local UTC offset.
- `image_path`: path to saved cropped face.
- `track_id`: track instance ID for this run.

Quick inspection:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/visitors.db'); print('Visitors:', c.execute('select count(*) from visitors').fetchone()[0]); print('Events:', c.execute('select event_id,face_id,event_type,timestamp,image_path,track_id from events order by event_id desc limit 20').fetchall()); c.close()"
```

## 9. Expected output

- A live window with face boxes, `face_id` / track ID, and unique visitor count.
- `logs/events.log`.
- Cropped images in date-organized entry/exit folders.
- `data/visitors.db` with registered identities and event rows.

Do not claim an accuracy percentage unless you actually measure it. Record test conditions, false matches, missed detections, and duplicate registrations in your report.

## 10. Known limitations and assumptions

1. YOLO and InsightFace use separate detection pipelines; the prototype associates their boxes by IoU. If the boxes do not overlap enough, that detection is skipped rather than assigning a potentially wrong embedding.
2. The tracker is a lightweight greedy IoU tracker, not DeepSORT/ByteTrack. It may lose identity when people cross, turn, leave and re-enter, or the camera moves.
3. Face embeddings can be affected by lighting, pose, masks, occlusion, and image quality.
4. A person who leaves and later returns should reuse the same `face_id` if the embedding similarity passes the configured threshold; that new appearance gets a new track and new entry/exit events.
5. Tracks still active at video end or manual stop are logged as exits so each created track has a closing event for that run.
6. `max_lost_detection_cycles` is counted in detection cycles, not raw video frames.
7. The unique count is a count of registered face identities, not a verified count of distinct human beings. False matches and duplicate registrations are possible.
8. The app uses local machine time for event timestamps.
9. No UI beyond the OpenCV display is required by the problem statement.

## 11. Test checklist

- [ ] Application starts with the sample video.
- [ ] A first-seen face is registered once.
- [ ] The same visible face is tracked across frames.
- [ ] Repeated frames do not create repeated entry events for the same track.
- [ ] A track disappearance produces one exit event.
- [ ] Entry and exit crops exist on disk.
- [ ] Event rows exist in SQLite and point to the image files.
- [ ] Returning face reuses its identity where matching is reliable.
- [ ] Unique count equals the number of rows in `visitors`.
- [ ] Detection skipping changes when detection runs.
- [ ] `logs/events.log` contains registration, recognition, entry, exit, and summary lines.
- [ ] README, planning document, architecture diagram, sample outputs, and demo video are included.

## 12. Submission checklist

1. Commit source code and documentation to GitHub.
2. Do **not** commit private video, real visitor face crops, database files, model weights with unclear redistribution rights, camera credentials, or personal biometric data. Provide only permitted synthetic/sample outputs or a sanitized demo.
3. Include screenshots of logs and database rows using the provided test video, while protecting identities.
4. Record and add a Loom/YouTube demo link to this README.
5. Submit the GitHub repository in the hackathon's Google Form before **12 PM, Monday, October 19, 2026**, according to the provided statement. Confirm the official form and deadline with the organizers.
6. Add the required line at the bottom of the README.

## 13. Demo video outline

1. Introduce the problem and architecture (30 seconds).
2. Show config and start command (20 seconds).
3. Run the sample video and show detection, tracking, and unique count (60 seconds).
4. Show `events.log`, entry/exit images, and SQLite visitor/event rows (60 seconds).
5. Explain assumptions, limitations, and CPU/GPU usage (30 seconds).

## 14. Compute estimate

Actual consumption depends on video resolution, frame rate, model weights, hardware, and the detection-skip setting. Measure on your test machine rather than presenting these as benchmark results.

- **CPU:** video decode, OpenCV crop/drawing, SQLite, logging, and Python orchestration always use CPU.
- **GPU:** YOLO can use a CUDA GPU if supported by the installed PyTorch build; InsightFace uses the ONNX Runtime providers available in the environment. This starter detects ONNX providers but does not force a particular GPU setup.
- **Memory:** model weights, decoded frames, and embeddings remain in RAM. Higher resolution and larger model packs generally use more memory.
- **Throughput:** skipping frames reduces detection/embedding work but can make short visits easier to miss. Record average FPS and CPU/GPU memory usage during your demo.

This project is a part of a hackathon run by https://katomaran.com
