from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Self, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ExperimentStore:
    """Append-oriented SQLite store for every scientific object and decision."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL, id TEXT NOT NULL, experiment_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            payload TEXT NOT NULL, PRIMARY KEY(kind, id))"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL,
            event_type TEXT NOT NULL, object_id TEXT, payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
        )
        self.connection.commit()

    def put(
        self, kind: str, value: BaseModel, created_at: datetime | str | None = None
    ) -> None:
        payload = value.model_dump_json()
        experiment_id = getattr(value, "experiment_id", None)
        timestamp = created_at or getattr(value, "created_at", None)
        timestamp_text = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        with self.connection:
            if timestamp_text is None:
                self.connection.execute(
                    "INSERT OR REPLACE INTO records(kind,id,experiment_id,payload) VALUES(?,?,?,?)",
                    (kind, value.id, experiment_id, payload),
                )
                self.connection.execute(
                    "INSERT INTO events(experiment_id,event_type,object_id,payload) VALUES(?,?,?,?)",
                    (experiment_id or value.id, f"{kind}.stored", value.id, "{}"),
                )
            else:
                self.connection.execute(
                    """INSERT OR REPLACE INTO records
                    (kind,id,experiment_id,created_at,payload) VALUES(?,?,?,?,?)""",
                    (kind, value.id, experiment_id, timestamp_text, payload),
                )
                self.connection.execute(
                    """INSERT INTO events
                    (experiment_id,event_type,object_id,payload,created_at) VALUES(?,?,?,?,?)""",
                    (
                        experiment_id or value.id,
                        f"{kind}.stored",
                        value.id,
                        "{}",
                        timestamp_text,
                    ),
                )

    def get(self, kind: str, object_id: str, model: type[T]) -> T:
        row = self.connection.execute(
            "SELECT payload FROM records WHERE kind=? AND id=?", (kind, object_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"missing {kind}/{object_id}")
        return model.model_validate_json(row[0])

    def list(self, kind: str, model: type[T], experiment_id: str | None = None) -> list[T]:
        if experiment_id is None:
            rows = self.connection.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY created_at,id", (kind,)
            )
        else:
            rows = self.connection.execute(
                "SELECT payload FROM records WHERE kind=? AND experiment_id=? ORDER BY created_at,id",
                (kind, experiment_id),
            )
        return [model.model_validate_json(row[0]) for row in rows]

    def event(
        self, experiment_id: str, event_type: str, payload: dict, object_id: str | None = None
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO events(experiment_id,event_type,object_id,payload) VALUES(?,?,?,?)",
                (experiment_id, event_type, object_id, json.dumps(payload, sort_keys=True)),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
