from pathlib import Path

from llm.storage import SQLITE_BUSY_TIMEOUT_MS, ConversationStore


def test_get_messages_limit_returns_recent_window_in_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "app.sqlite3")
    conversation = store.create_conversation(user_id="user-a")

    for index in range(1, 6):
        store.add_message(
            user_id="user-a",
            conversation_id=conversation.id,
            role="user",
            content=f"u{index}",
            status="done",
        )
        store.add_message(
            user_id="user-a",
            conversation_id=conversation.id,
            role="assistant",
            content=f"a{index}",
            status="done",
        )

    recent_messages = store.get_messages(user_id="user-a", conversation_id=conversation.id, limit=4)

    assert [(item.role, item.content) for item in recent_messages] == [
        ("user", "u4"),
        ("assistant", "a4"),
        ("user", "u5"),
        ("assistant", "a5"),
    ]


def test_store_enables_sqlite_write_pragmas(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "app.sqlite3")

    with store._connect() as connection:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
        busy_timeout = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])

    assert journal_mode == "wal"
    assert synchronous == 1
    assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS
