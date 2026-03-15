"""
db/mongo.py
-----------
MongoDB persistence for analysis history and operational records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    from pymongo.server_api import ServerApi
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False
    MongoClient = None
    PyMongoError = Exception
    ServerApi = None

from src.config import MONGODB_CONFIG


class MongoPersistence:
    def __init__(self) -> None:
        self._client = None
        self._db = None
        self._enabled = bool(MONGODB_CONFIG["uri"]) and _PYMONGO_AVAILABLE

    @property
    def enabled(self) -> bool:
        return self._enabled

    def connect(self) -> None:
        if not self._enabled or self._client is not None:
            return
        try:
            self._client = MongoClient(
                MONGODB_CONFIG["uri"],
                server_api=ServerApi("1") if ServerApi is not None else None,
                serverSelectionTimeoutMS=MONGODB_CONFIG["connect_timeout_ms"],
            )
            self._client.admin.command("ping")
            self._db = self._client[MONGODB_CONFIG["database"]]
            logger.info("MongoDB connected | database={}", MONGODB_CONFIG["database"])
        except PyMongoError as exc:
            logger.warning("MongoDB connection unavailable: {}", exc)
            self._client = None
            self._db = None

    def health_status(self) -> str:
        if not self._enabled:
            return "disabled"
        self.connect()
        return "ready" if self._db is not None else "unavailable"

    def save_analysis(self, kind: str, payload: dict[str, Any]) -> Optional[str]:
        if not self._enabled:
            return None
        self.connect()
        if self._db is None:
            return None
        collection_name = MONGODB_CONFIG["collections"].get(kind, "analysis_history")
        document = {
            **payload,
            "kind": kind,
            "created_at": datetime.now(timezone.utc),
        }
        result = self._db[collection_name].insert_one(document)
        return str(result.inserted_id)

    def recent_records(self, kind: str = "analyses", limit: int = 20) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        self.connect()
        if self._db is None:
            return []
        collection_name = MONGODB_CONFIG["collections"].get(kind, "analysis_history")
        items = list(
            self._db[collection_name]
            .find({}, {"_id": 1, "created_at": 1, "query_id": 1, "filename": 1, "possible_conditions": 1})
            .sort("created_at", -1)
            .limit(limit)
        )
        for item in items:
            item["_id"] = str(item["_id"])
        return items


mongo_persistence = MongoPersistence()
