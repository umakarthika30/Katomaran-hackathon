from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from .face_pipeline import box_iou

@dataclass
class Track:
    track_id: int
    face_id: str
    bbox: list
    crop: Any
    missed_cycles: int = 0

class IoUTracker:
    """Lightweight greedy IoU tracker. Misses are counted per detection cycle."""
    def __init__(self, iou_threshold: float, max_lost_cycles: int):
        self.iou_threshold = float(iou_threshold)
        self.max_lost_cycles = int(max_lost_cycles)
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def update(self, detections: List[dict]) -> Tuple[List[Track], List[Track]]:
        for track in self.tracks.values():
            track.missed_cycles += 1

        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = box_iou(track.bbox, detection["bbox"])
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))
        candidates.sort(reverse=True, key=lambda item: item[0])

        matched_tracks, matched_detections = set(), set()
        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_index]
            track.bbox = detection["bbox"]
            track.crop = detection["crop"]
            track.missed_cycles = 0
            # Keep the identity assigned when this track was created.
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        entered = []
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track = Track(
                track_id=self.next_track_id,
                face_id=detection["face_id"],
                bbox=detection["bbox"],
                crop=detection["crop"]
            )
            self.tracks[self.next_track_id] = track
            entered.append(track)
            self.next_track_id += 1

        exited = []
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if track.missed_cycles > self.max_lost_cycles:
                exited.append(track)
                del self.tracks[track_id]

        return entered, exited

    def active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def close_all(self) -> List[Track]:
        remaining = list(self.tracks.values())
        self.tracks.clear()
        return remaining
