import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional


class HistoryRepository:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    top_species TEXT,
                    confidence REAL,
                    is_bird INTEGER NOT NULL,
                    reasoning TEXT,
                    result_json TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            conn.commit()

    def add_record(
        self,
        image_hash: str,
        file_name: str,
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> None:
        created_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        top_species = ""
        confidence = 0.0
        is_bird = 0
        reasoning = ""
        result_json = "{}"

        if result:
            predictions = result.get("predictions") or []
            if predictions:
                top_species = predictions[0].get("common_name") or predictions[0].get("species") or ""
                confidence = float(predictions[0].get("confidence", 0.0))
            is_bird = 1 if result.get("is_bird", True) else 0
            reasoning = str(result.get("reasoning", ""))
            result_json = json.dumps(result, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO history(created_at, image_hash, file_name, top_species, confidence, is_bird, reasoning, result_json, error)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, image_hash, file_name, top_species, confidence, is_bird, reasoning, result_json, error),
            )
            conn.commit()

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, file_name, top_species, confidence, is_bird, reasoning, error
                FROM history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "created_at": row[0],
                    "file_name": row[1],
                    "top_species": row[2],
                    "confidence": row[3],
                    "is_bird": bool(row[4]),
                    "reasoning": row[5],
                    "error": row[6],
                }
            )
        return result

    def get_latest_success_by_hash(self, image_hash: str) -> Optional[Dict[str, Any]]:
        if not image_hash:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT result_json
                FROM history
                WHERE image_hash = ? AND error = '' AND result_json <> '{}'
                ORDER BY id DESC
                LIMIT 1
                """,
                (image_hash,),
            ).fetchone()

        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
