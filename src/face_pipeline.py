from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0

class FacePipeline:
    """
    YOLO provides face boxes. InsightFace supplies facial embeddings.
    The two detector outputs are associated by bounding-box IoU.
    """
    def __init__(self, config: dict, logger):
        model_path = Path(config["yolo_model_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO face weights not found: {model_path}. "
                "Place a compatible face-detection .pt file at this path."
            )
        self.logger = logger
        self.confidence = float(config["confidence_threshold"])
        self.yolo = YOLO(str(model_path))

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:
            providers = ["CPUExecutionProvider"]

        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.face_app = FaceAnalysis(
            name=config.get("insightface_model", "buffalo_l"),
            providers=providers
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.logger.info("Models loaded. ONNX providers: %s", providers)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        result = self.yolo.predict(
            source=frame, conf=self.confidence, verbose=False, imgsz=640
        )[0]
        yolo_boxes = []
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(x1, frame.shape[1] - 1))
                x2 = max(0, min(x2, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                y2 = max(0, min(y2, frame.shape[0]))
                if x2 > x1 and y2 > y1:
                    yolo_boxes.append([x1, y1, x2, y2])

        if not yolo_boxes:
            return []

        insight_faces = self.face_app.get(frame)
        used = set()
        detections = []

        for bbox in yolo_boxes:
            best_index, best_iou = None, 0.0
            for index, face in enumerate(insight_faces):
                if index in used:
                    continue
                overlap = box_iou(bbox, face.bbox)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap

            # Avoid assigning an embedding from a different nearby face.
            if best_index is None or best_iou < 0.10:
                self.logger.warning(
                    "YOLO detected a face but no matching InsightFace embedding was found."
                )
                continue

            used.add(best_index)
            face = insight_faces[best_index]
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2].copy()
            embedding = np.asarray(face.embedding, dtype=np.float32)
            if crop.size and embedding.size:
                detections.append({
                    "bbox": bbox,
                    "crop": crop,
                    "embedding": embedding
                })
        return detections
