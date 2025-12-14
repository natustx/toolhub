"""SQLite-backed storage for structured API operations from OpenAPI specs."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from toolhub.paths import get_toolhub_home

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def get_operations_db_path() -> Path:
    """Get path to the operations SQLite database."""
    return get_toolhub_home() / "operations.db"


@dataclass
class Operation:
    """A single API operation extracted from an OpenAPI spec."""

    tool_id: str
    operation_id: str
    method: str  # GET, POST, etc.
    path: str  # /users/{id}
    summary: str
    description: str
    tags: list[str]
    parameters: list[dict]  # Simplified parameter info
    request_body: dict | None  # Content type and schema info
    responses: dict  # Status code -> description

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_id": self.tool_id,
            "operation_id": self.operation_id,
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
            "request_body": self.request_body,
            "responses": self.responses,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Operation:
        """Create from database row."""
        return cls(
            tool_id=row["tool_id"],
            operation_id=row["operation_id"],
            method=row["method"],
            path=row["path"],
            summary=row["summary"] or "",
            description=row["description"] or "",
            tags=json.loads(row["tags"]) if row["tags"] else [],
            parameters=json.loads(row["parameters"]) if row["parameters"] else [],
            request_body=json.loads(row["request_body"]) if row["request_body"] else None,
            responses=json.loads(row["responses"]) if row["responses"] else {},
        )


class OperationsStore:
    """SQLite-backed store for API operations from OpenAPI specs."""

    def __init__(self, db_path: Path | None = None):
        """Initialize operations store.

        Args:
            db_path: Optional custom database path
        """
        self.db_path = db_path or get_operations_db_path()
        self._conn: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        """Ensure database connection and schema exist."""
        if self._conn is not None:
            return self._conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Create tables if needed
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                summary TEXT,
                description TEXT,
                tags TEXT,  -- JSON array
                parameters TEXT,  -- JSON array
                request_body TEXT,  -- JSON object
                responses TEXT,  -- JSON object
                UNIQUE(tool_id, operation_id)
            );

            CREATE INDEX IF NOT EXISTS idx_operations_tool_id ON operations(tool_id);
            CREATE INDEX IF NOT EXISTS idx_operations_method ON operations(method);
            CREATE INDEX IF NOT EXISTS idx_operations_path ON operations(path);

            -- Full-text search for summary and description
            CREATE VIRTUAL TABLE IF NOT EXISTS operations_fts USING fts5(
                operation_id,
                summary,
                description,
                path,
                content='operations',
                content_rowid='id'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS operations_ai AFTER INSERT ON operations BEGIN
                INSERT INTO operations_fts(rowid, operation_id, summary, description, path)
                VALUES (new.id, new.operation_id, new.summary, new.description, new.path);
            END;

            CREATE TRIGGER IF NOT EXISTS operations_ad AFTER DELETE ON operations BEGIN
                INSERT INTO operations_fts(
                    operations_fts, rowid, operation_id, summary, description, path
                ) VALUES (
                    'delete', old.id, old.operation_id, old.summary, old.description, old.path
                );
            END;

            CREATE TRIGGER IF NOT EXISTS operations_au AFTER UPDATE ON operations BEGIN
                INSERT INTO operations_fts(
                    operations_fts, rowid, operation_id, summary, description, path
                ) VALUES (
                    'delete', old.id, old.operation_id, old.summary, old.description, old.path
                );
                INSERT INTO operations_fts(rowid, operation_id, summary, description, path)
                VALUES (new.id, new.operation_id, new.summary, new.description, new.path);
            END;
        """)
        self._conn.commit()

        return self._conn

    def add_operations(self, operations: list[Operation]) -> int:
        """Add operations to the store.

        Args:
            operations: List of Operation objects to add

        Returns:
            Number of operations added
        """
        if not operations:
            return 0

        conn = self._ensure_db()

        added = 0
        for op in operations:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO operations
                    (tool_id, operation_id, method, path, summary, description,
                     tags, parameters, request_body, responses)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        op.tool_id,
                        op.operation_id,
                        op.method,
                        op.path,
                        op.summary,
                        op.description,
                        json.dumps(op.tags),
                        json.dumps(op.parameters),
                        json.dumps(op.request_body) if op.request_body else None,
                        json.dumps(op.responses),
                    ),
                )
                added += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to add operation {op.operation_id}: {e}")

        conn.commit()
        logger.info(f"Added {added} operations")
        return added

    def search(
        self,
        query: str,
        tool_id: str | None = None,
        method: str | None = None,
        limit: int = 10,
    ) -> list[Operation]:
        """Search operations using full-text search.

        Args:
            query: Search query (matched against summary, description, path)
            tool_id: Optional filter by tool
            method: Optional filter by HTTP method
            limit: Maximum results

        Returns:
            List of matching operations
        """
        conn = self._ensure_db()

        # Build query
        sql = """
            SELECT o.*
            FROM operations o
            JOIN operations_fts fts ON o.id = fts.rowid
            WHERE operations_fts MATCH ?
        """
        params: list = [query]

        if tool_id:
            sql += " AND o.tool_id = ?"
            params.append(tool_id)

        if method:
            sql += " AND o.method = ?"
            params.append(method.upper())

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [Operation.from_row(row) for row in rows]

    def search_by_path(
        self,
        path_pattern: str,
        tool_id: str | None = None,
    ) -> list[Operation]:
        """Search operations by path pattern.

        Args:
            path_pattern: Path pattern (supports % wildcards)
            tool_id: Optional filter by tool

        Returns:
            List of matching operations
        """
        conn = self._ensure_db()

        sql = "SELECT * FROM operations WHERE path LIKE ?"
        params: list = [path_pattern]

        if tool_id:
            sql += " AND tool_id = ?"
            params.append(tool_id)

        rows = conn.execute(sql, params).fetchall()
        return [Operation.from_row(row) for row in rows]

    def get_operations_for_tool(self, tool_id: str) -> list[Operation]:
        """Get all operations for a tool.

        Args:
            tool_id: Tool identifier

        Returns:
            List of operations
        """
        conn = self._ensure_db()
        rows = conn.execute(
            "SELECT * FROM operations WHERE tool_id = ? ORDER BY path, method",
            (tool_id,),
        ).fetchall()
        return [Operation.from_row(row) for row in rows]

    def delete_tool_operations(self, tool_id: str) -> int:
        """Delete all operations for a tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Number of operations deleted
        """
        conn = self._ensure_db()
        cursor = conn.execute("DELETE FROM operations WHERE tool_id = ?", (tool_id,))
        conn.commit()
        deleted = cursor.rowcount
        logger.info(f"Deleted {deleted} operations for {tool_id}")
        return deleted

    def count(self, tool_id: str | None = None) -> int:
        """Count operations.

        Args:
            tool_id: Optional filter by tool

        Returns:
            Number of operations
        """
        conn = self._ensure_db()

        if tool_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM operations WHERE tool_id = ?", (tool_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM operations").fetchone()

        return row[0] if row else 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
