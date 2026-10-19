import json
import sys
from datetime import datetime
from pathlib import Path
import cv2

from src.database import VisitorDatabase
from src.face_pipeline import FacePipeline
from src.logger_setup import configure_logger
from src.tracker import IoUTracker

def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def resolve_video_source(source):
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text

def save_event_image(crop, logs_root: str, event_type: str,
                     face_id: str, track_id: int, timestamp: str) -> str:
    day = datetime.now().strftime("%Y-%m-%d")
    folder_name = "entries" if event_type == "entry" else "exits"
    folder = Path(logs_root) / folder_name / day
    folder.mkdir(parents=True, exist_ok=True)
    safe_time = timestamp.replace(":", "-").replace("+", "_")
    path = folder / f"{safe_time}_{face_id}_track{track_id}.jpg"
    if crop is None or crop.size == 0:
        raise ValueError("Cannot save an empty face crop")
    if not cv2.imwrite(str(path), crop):
        raise IOError(f"Could not write event image: {path}")
    return str(path)

def record_event(db, logger, track, event_type: str, logs_root: str):
    timestamp = timestamp_now()
    image_path = save_event_image(
        track.crop, logs_root, event_type, track.face_id, track.track_id, timestamp
    )
    db.add_event(track.face_id, event_type, timestamp, image_path, track.track_id)
    logger.info(
        "FACE_%s | face_id=%s | track_id=%s | image=%s",
        event_type.upper(), track.face_id, track.track_id, image_path
    )

def main():
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Run the program from the project root.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    logger = configure_logger()
    db = VisitorDatabase(config["database_path"])
    pipeline = FacePipeline(config, logger)
    tracker = IoUTracker(
        config["tracking_iou_threshold"],
        config["max_lost_detection_cycles"]
    )

    source = resolve_video_source(config["video_source"])
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source {source!r}. Check sample.mp4 or the RTSP URL."
        )

    skip = max(0, int(config["detection_skip_frames"]))
    detection_interval = skip + 1
    frame_index = 0
    logger.info("Stream started. source=%s, detection_interval=%s", source, detection_interval)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.info("Stream ended or frame could not be read.")
                break

            if frame_index % detection_interval == 0:
                raw_detections = pipeline.detect(frame)
                detections = []
                for item in raw_detections:
                    face_id, is_new, similarity = db.identify_or_register(
                        item["embedding"], float(config["face_match_threshold"])
                    )
                    item["face_id"] = face_id
                    detections.append(item)
                    if is_new:
                        logger.info(
                            "EMBEDDING_REGISTERED | face_id=%s | previous_best_similarity=%.3f",
                            face_id, similarity
                        )
                    else:
                        logger.info(
                            "FACE_RECOGNISED | face_id=%s | similarity=%.3f",
                            face_id, similarity
                        )

                entered, exited = tracker.update(detections)
                for track in entered:
                    record_event(db, logger, track, "entry", config["logs_root"])
                for track in exited:
                    record_event(db, logger, track, "exit", config["logs_root"])

            for track in tracker.active_tracks():
                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{track.face_id} / T{track.track_id}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2
                )

            count_text = f"Unique visitors: {db.unique_visitor_count()}"
            cv2.putText(frame, count_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow(config.get("window_name", "Face Tracker"), frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("Stop requested by user.")
                break
    finally:
        # Treat tracks still present at shutdown/end-of-file as exits for this run.
        for track in tracker.close_all():
            try:
                record_event(db, logger, track, "exit", config["logs_root"])
            except Exception:
                logger.exception("Failed to write final exit event for track %s", track.track_id)
        capture.release()
        cv2.destroyAllWindows()
        logger.info(
            "SUMMARY | unique_visitors=%s | total_events=%s",
            db.unique_visitor_count(), db.event_count()
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger = configure_logger()
        logger.exception("Application stopped with error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
