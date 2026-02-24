import os
import sqlite3
from datetime import datetime


class ChatHistoryStore:
    def __init__(self):
        self.db_path = None
        self.conn = None

    def connect(self, db_path: str = None):
        if db_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base, "data", "chat_history.db")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"[ChatHistory] Connected — DB: {db_path}")

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Migrate: drop old chat_messages table if it has the old session_id schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(chat_messages)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'session_id' in columns or 'conversation_id' not in columns:
                print("[ChatHistory] Migrating: dropping old chat_messages table...")
                cursor.execute("DROP TABLE chat_messages")
                cursor.execute("DROP INDEX IF EXISTS idx_session")
                cursor.execute("DROP INDEX IF EXISTS idx_msg_conv")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'محادثة جديدة',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_user
            ON conversations(user_id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_conv
            ON chat_messages(conversation_id)
        """)
        self.conn.commit()

    # --- Conversation management ---

    def create_conversation(self, conversation_id: str, user_id: str, title: str = "محادثة جديدة") -> dict:
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, title, now, now)
        )
        self.conn.commit()
        return {"id": conversation_id, "user_id": user_id, "title": title, "created_at": now}

    def list_conversations(self, user_id: str) -> list[dict]:
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        return [
            {"id": row["id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"]}
            for row in rows
        ]

    def update_conversation_title(self, conversation_id: str, title: str):
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.utcnow().isoformat(), conversation_id)
        )
        self.conn.commit()

    def delete_conversation(self, conversation_id: str):
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()
        print(f"[ChatHistory] Deleted conversation: {conversation_id}")

    # --- Message management ---

    def add_message(self, conversation_id: str, role: str, content: str):
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content)
        )
        # Update conversation timestamp
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), conversation_id)
        )
        self.conn.commit()

    def get_history(self, conversation_id: str, limit: int = 50) -> list[dict]:
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT role, content, created_at FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit)
        )
        rows = cursor.fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def clear_history(self, conversation_id: str):
        if self.conn is None:
            raise RuntimeError("ChatHistory not connected.")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        self.conn.commit()

    def disconnect(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        print("[ChatHistory] Disconnected.")
