"""HTTP client for communicating with the toolhub daemon.

This module provides the client interface for talking to toolhubd.
When the daemon is not running, CLI commands work directly.
When the daemon is running, commands are proxied through it for
better performance (warm embeddings, persistent connections).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import httpx

from toolhub.config import load_config
from toolhub.paths import get_pid_path

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT = 30  # seconds to wait for daemon to start
STARTUP_CHECK_INTERVAL = 0.2  # seconds between health checks


def get_daemon_url() -> str:
    """Get the daemon URL from config."""
    config = load_config()
    return f"http://{config.daemon.host}:{config.daemon.port}"


def is_daemon_running() -> bool:
    """Check if the daemon is running.

    Returns:
        True if daemon is responding to health checks
    """
    pid_path = get_pid_path()
    if not pid_path.exists():
        return False

    try:
        url = get_daemon_url()
        response = httpx.get(f"{url}/health", timeout=1.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def start_daemon() -> bool:
    """Start the daemon in the background.

    Returns:
        True if daemon started successfully
    """
    logger.info("Starting daemon...")

    # Use the same Python interpreter to run the daemon module
    python = sys.executable
    subprocess.Popen(
        [python, "-m", "toolhub.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for daemon to become healthy
    start_time = time.time()
    while time.time() - start_time < STARTUP_TIMEOUT:
        if is_daemon_running():
            logger.info("Daemon started successfully")
            return True
        time.sleep(STARTUP_CHECK_INTERVAL)

    logger.error("Daemon failed to start within timeout")
    return False


def stop_daemon() -> bool:
    """Stop the running daemon.

    Returns:
        True if daemon was stopped
    """
    if not is_daemon_running():
        return False

    try:
        client = DaemonClient()
        client.stop()
        client.close()

        # Wait for daemon to stop
        start_time = time.time()
        while time.time() - start_time < 5:
            if not is_daemon_running():
                return True
            time.sleep(0.2)

        # Force kill if still running
        pid_path = get_pid_path()
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            pid_path.unlink(missing_ok=True)

        return True
    except Exception as e:
        logger.error(f"Failed to stop daemon: {e}")
        return False


def ensure_daemon() -> bool:
    """Ensure the daemon is running, starting it if needed.

    This is the lazy start logic - automatically starts the daemon
    on first query if not already running.

    Returns:
        True if daemon is now running
    """
    if is_daemon_running():
        return True

    return start_daemon()


class DaemonClient:
    """Client for the toolhub daemon API."""

    def __init__(self, base_url: str | None = None, auto_start: bool = False):
        """Initialize the daemon client.

        Args:
            base_url: Override the daemon URL
            auto_start: If True, automatically start daemon if not running
        """
        self.base_url = base_url or get_daemon_url()
        self.auto_start = auto_start
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            if self.auto_start:
                ensure_daemon()
            self._client = httpx.Client(base_url=self.base_url, timeout=30.0)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def health(self) -> dict:
        """Check daemon health."""
        response = self._get_client().get("/health")
        response.raise_for_status()
        return response.json()

    def add_tool(
        self,
        url: str,
        name: str | None = None,
        replace: bool = False,
    ) -> dict:
        """Add a documentation source."""
        response = self._get_client().post(
            "/tools/add",
            json={"url": url, "name": name, "replace": replace},
        )
        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        tool_ids: list[str] | None = None,
        limit: int = 5,
    ) -> dict:
        """Search documentation."""
        response = self._get_client().post(
            "/tools/query",
            json={"query": query, "tool_ids": tool_ids, "limit": limit},
        )
        response.raise_for_status()
        return response.json()

    def list_tools(self) -> dict:
        """List all indexed tools."""
        response = self._get_client().get("/tools")
        response.raise_for_status()
        return response.json()

    def get_tool(self, tool_id: str) -> dict:
        """Get tool details."""
        response = self._get_client().get(f"/tools/{tool_id}")
        response.raise_for_status()
        return response.json()

    def remove_tool(self, tool_id: str) -> dict:
        """Remove a tool."""
        response = self._get_client().delete(f"/tools/{tool_id}")
        response.raise_for_status()
        return response.json()

    def stop(self) -> dict:
        """Stop the daemon."""
        response = self._get_client().post("/stop")
        response.raise_for_status()
        return response.json()
