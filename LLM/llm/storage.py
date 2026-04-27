from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from llm.models import Citation, ConversationSummary, StoredMessage


DEFAULT_CONVERSATION_TITLE = "新会话"


class ConversationNotFoundError(LookupError):
    """Raised when a conversation is missing or does not belong to the user."""


class ConversationStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def list_conversations(self, user_id: str) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at, last_message_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY last_message_at DESC, created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_conversation(row) for row in rows]

    def create_conversation(self, user_id: str, title: str | None = None) -> ConversationSummary:
        conversation_id = str(uuid.uuid4())
        timestamp = _utc_now()
        normalized_title = _normalize_title(title) or DEFAULT_CONVERSATION_TITLE
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, user_id, title, created_at, updated_at, last_message_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_id, normalized_title, timestamp, timestamp, timestamp),
            )
            row = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at, last_message_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
        return self._row_to_conversation(row)

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> ConversationSummary:
        normalized_title = _normalize_title(title)
        if normalized_title is None:
            raise ValueError("title must not be empty")
        updated_at = _utc_now()
        with self._connect() as connection:
            self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (normalized_title, updated_at, conversation_id, user_id),
            )
            row = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at, last_message_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
        return self._row_to_conversation(row)

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        with self._connect() as connection:
            self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
            connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )

    def get_messages(self, user_id: str, conversation_id: str) -> list[StoredMessage]:
        with self._connect() as connection:
            self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, status, model, retrieval_count, citations_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def ensure_conversation(self, user_id: str, conversation_id: str) -> ConversationSummary:
        with self._connect() as connection:
            row = self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
        return self._row_to_conversation(row)

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        status: str,
        citations: list[Citation] | None = None,
        model: str | None = None,
        retrieval_count: int | None = None,
    ) -> StoredMessage:
        message_id = str(uuid.uuid4())
        created_at = _utc_now()
        citations_json = _serialize_citations(citations or [])
        with self._connect() as connection:
            conversation = self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, status, model, retrieval_count, citations_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    status,
                    model,
                    retrieval_count,
                    citations_json,
                    created_at,
                ),
            )
            next_title = conversation["title"]
            if role == "user" and conversation["title"] == DEFAULT_CONVERSATION_TITLE:
                next_title = default_conversation_title(content)
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?, last_message_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (next_title, created_at, created_at, conversation_id, user_id),
            )
            row = connection.execute(
                """
                SELECT id, conversation_id, role, content, status, model, retrieval_count, citations_json, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return self._row_to_message(row)

    def update_message(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        *,
        content: str,
        status: str,
        citations: list[Citation] | None = None,
        model: str | None = None,
        retrieval_count: int | None = None,
    ) -> StoredMessage:
        updated_at = _utc_now()
        citations_json = _serialize_citations(citations or [])
        with self._connect() as connection:
            self._require_conversation(connection, user_id=user_id, conversation_id=conversation_id)
            row = connection.execute(
                """
                SELECT id FROM messages
                WHERE id = ? AND conversation_id = ?
                """,
                (message_id, conversation_id),
            ).fetchone()
            if row is None:
                raise LookupError(f"Message '{message_id}' was not found.")
            connection.execute(
                """
                UPDATE messages
                SET content = ?, status = ?, model = ?, retrieval_count = ?, citations_json = ?
                WHERE id = ? AND conversation_id = ?
                """,
                (content, status, model, retrieval_count, citations_json, message_id, conversation_id),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?, last_message_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (updated_at, updated_at, conversation_id, user_id),
            )
            updated_row = connection.execute(
                """
                SELECT id, conversation_id, role, content, status, model, retrieval_count, citations_json, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return self._row_to_message(updated_row)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT,
                    retrieval_count INTEGER,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_last_message "
                "ON conversations(user_id, last_message_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created_at "
                "ON messages(conversation_id, created_at ASC)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _require_conversation(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, last_message_at
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found for user '{user_id}'.")
        return row

    @staticmethod
    def _row_to_conversation(row: sqlite3.Row) -> ConversationSummary:
        return ConversationSummary(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_message_at=str(row["last_message_at"]),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> StoredMessage:
        citations_payload = json.loads(str(row["citations_json"]))
        return StoredMessage(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            status=str(row["status"]),
            model=row["model"],
            retrieval_count=row["retrieval_count"],
            citations=[Citation(**item) for item in citations_payload],
            created_at=str(row["created_at"]),
        )


def default_conversation_title(question: str) -> str:
    trimmed = " ".join(question.strip().split())
    if not trimmed:
        return DEFAULT_CONVERSATION_TITLE
    return trimmed[:30]


def _normalize_title(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = " ".join(value.strip().split())
    return trimmed or None


def _serialize_citations(citations: list[Citation]) -> str:
    return json.dumps([item.to_dict() for item in citations], ensure_ascii=False)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
