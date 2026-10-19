# Interview Questions and Suggested Answers

Use these as learning notes. Explain the implementation in your own words and be honest about what you have actually tested.

## 1. What problem does your application solve?
It detects and tracks faces in a video stream, assigns a stable database identity to a face embedding, counts registered identities, and records entry/exit events with timestamps and face crops.

## 2. Why use YOLO?
YOLO performs object detection in a single forward pass and is commonly used for real-time detection. This project requires a **face-specific YOLO weights file**; a generic object model is not automatically a face detector.

## 3. Why use InsightFace?
InsightFace provides face-analysis models that can produce embeddings. An embedding is a numeric representation of facial features that can be compared with stored embeddings.

## 4. What is a face embedding?
A vector of numbers representing features extracted from a face image. The application normalizes it and compares it with stored vectors.

## 5. How do you compare two faces?
The prototype uses cosine similarity between normalized embeddings. A score above `face_match_threshold` is treated as a match. The threshold must be tuned and evaluated; it is not universally correct.

## 6. How do you avoid counting the same person repeatedly?
When an embedding matches a stored identity, the existing `face_id` is reused. The unique visitor count is the number of rows in the `visitors` table, not the number of detections or events.

## 7. What is the difference between visitor identity and track ID?
`face_id` represents a re-identified face across appearances. `track_id` represents one continuous appearance/track during the current run. The same face can have different track IDs if it leaves and later returns.

## 8. How does tracking work?
The prototype uses greedy Intersection over Union (IoU) matching between current and previous bounding boxes. A track is marked missed when not matched and is closed after `max_lost_detection_cycles`.

## 9. What is IoU?
Intersection over Union is intersection area divided by union area for two boxes. A larger IoU indicates more overlap.

## 10. Why skip frames?
Running detection on every frame may consume more compute. Skipping frames can improve throughput, but it can miss short appearances or fast movement. We expose this trade-off in `config.json`.

## 11. Why SQLite?
SQLite is simple, local, transactional, and requires no separate database server. It fits a hackathon prototype. A multi-camera production system may need a server database.

## 12. What is stored in the database?
The `visitors` table stores the face ID, embedding, and first-seen timestamp. The `events` table stores event ID, face ID, entry/exit type, timestamp, image path, and track ID.

## 13. How do you ensure entry and exit are logged once?
An entry is written when a new track is created. An exit is written when that track exceeds the missed-cycle limit or when the program closes. The tracker removes a closed track, preventing repeated exit events for that track.

## 14. What happens if the same person returns later?
If the embedding similarity passes the configured threshold, the old `face_id` is reused. A new track gets a new track ID and its own entry/exit events, while the unique identity count remains unchanged.

## 15. What are the main limitations?
Face matching can fail with poor lighting, pose changes, masks, occlusion, or similar-looking faces. The simple IoU tracker can switch tracks when people cross. The separate YOLO and InsightFace detections can also disagree.

## 16. Why do you use both YOLO and InsightFace detection?
YOLO is the required detection component in the specification, while InsightFace supplies embeddings. This prototype matches the two systems' bounding boxes by IoU. It is a trade-off and a place for future optimization.

## 17. How would you improve it?
Use a well-tested face-specific detector, stronger multi-object tracking, batch embedding extraction, calibrated thresholds, persistent track state, asynchronous event writing, structured metrics, unit/integration tests, and a privacy-aware retention policy.

## 18. How would you test correctness?
Manually annotate a short video segment, compare expected unique identities with the database count, inspect false matches and duplicate registrations, check entry/exit pairs, verify image files, and query SQLite.

## 19. How do you estimate CPU/GPU usage?
Measure it on the target machine while processing a fixed clip. Record hardware, resolution, frame rate, detection interval, processing FPS, CPU utilization, RAM, GPU utilization, and VRAM. Do not present guesses as measured benchmarks.

## 20. Why are logs important?
They provide an audit trail for registrations, recognitions, tracking events, entry/exit records, and errors. The problem statement says log files are mandatory for verification.

## 21. What is the difference between detection and recognition?
Detection locates a face in an image. Recognition/re-identification compares the face's embedding to known embeddings to determine whether it matches an existing stored identity.

## 22. What if no InsightFace embedding matches a YOLO box?
This prototype logs a warning and skips the detection rather than associating a potentially incorrect embedding. A stronger implementation could use a single shared detection/embedding pipeline or more robust box association.

## 23. What privacy risks exist?
Face embeddings and face crops are sensitive biometric data. The demo should use authorized footage, restrict access, avoid uploading real faces to a public repository, set retention limits, and follow applicable law and organizer rules.

## 24. Why not use the `face_recognition` Python library?
The task explicitly asks to avoid it and recommends InsightFace/ArcFace or another state-of-the-art approach.

## 25. Explain your architecture in 60 seconds.
Video source → YOLO face boxes → InsightFace embeddings → SQLite identity matching → IoU tracking → entry/exit event manager → cropped images + SQLite event records + `events.log` → unique visitor count and live display.
