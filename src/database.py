import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import numpy as np

class VisitorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _create_tables(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    face_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    first_seen TEXT NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    track_id INTEGER NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def identify_or_register(
        self, embedding: np.ndarray, threshold: float
    ) -> Tuple[str, bool, float]:
        """Return (face_id, newly_registered, best_similarity)."""
        candidate = self._normalise(embedding)
        best_id: Optional[str] = None
        best_similarity = -1.0

        with self._connect() as con:
            rows = con.execute("SELECT face_id, embedding FROM visitors").fetchall()
            for face_id, blob in rows:
                stored = np.frombuffer(blob, dtype=np.float32)
                if stored.shape != candidate.shape:
                    continue
                similarity = float(np.dot(candidate, self._normalise(stored)))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = face_id

            if best_id is not None and best_similarity >= threshold:
                return best_id, False, best_similarity

            face_id = f"visitor_{uuid.uuid4().hex[:8]}"
            con.execute(
                "INSERT INTO visitors(face_id, embedding, first_seen) VALUES (?, ?, ?)",
                (face_id, candidate.tobytes(), datetime.now().astimezone().isoformat(timespec="seconds"))
            )
            return face_id, True, best_similarity

    def add_event(self, face_id: str, event_type: str, timestamp: str,
                  image_path: str, track_id: int) -> None:
        with self._connect() as con:
            con.execute("""
                INSERT INTO events(face_id, event_type, timestamp, image_path, track_id)
                VALUES (?, ?, ?, ?, ?)
            """, (face_id, event_type, timestamp, image_path, track_id))

    def unique_visitor_count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM visitors").fetchone()[0])

    def event_count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
