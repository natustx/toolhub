"""Path management and directory structure for toolhub.

Default storage location: ~/.toolhub/
Can be overridden via TOOLHUB_HOME environment variable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlparse


def get_toolhub_home() -> Path:
    """Get the toolhub home directory.

    Uses TOOLHUB_HOME env var if set, otherwise ~/.toolhub/
    """
    env_home = os.environ.get("TOOLHUB_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".toolhub"


def get_config_path() -> Path:
    """Get path to config.toml."""
    return get_toolhub_home() / "config.toml"


def get_sources_path() -> Path:
    """Get path to sources.json registry."""
    return get_toolhub_home() / "sources.json"


def get_cache_dir() -> Path:
    """Get path to cache directory for raw crawled markdown."""
    return get_toolhub_home() / "cache"


def get_tool_cache_dir(tool_id: str) -> Path:
    """Get cache directory for a specific tool."""
    return get_cache_dir() / tool_id


def _url_to_source_name(url: str) -> str:
    """Convert a URL to a human-readable source name.

    Examples:
        https://github.com/opentensor/bittensor -> opentensor-bittensor
        https://github.com/fastapi/fastapi -> fastapi-fastapi
        https://fastapi.tiangolo.com/llms.txt -> fastapi.tiangolo.com-llmstxt
        https://docs.bittensor.com -> docs.bittensor.com
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    # GitHub URLs: extract org-repo
    if "github.com" in host or "gitlab.com" in host:
        parts = path.split("/")
        if len(parts) >= 2:
            org, repo = parts[0], parts[1]
            # Remove .git suffix if present
            repo = repo.removesuffix(".git")
            return f"{org}-{repo}"

    # llms.txt files: domain-llmstxt
    if path.endswith("llms.txt") or "llms.txt" in path:
        return f"{host}-llmstxt"

    # Default: just use domain
    return host


def get_source_cache_dir(tool_id: str, url: str) -> Path:
    """Get cache directory for a specific source within a tool.

    Format: <tool_id>/<source_name>-<short_hash>/

    Examples:
        bittensor/opentensor-bittensor-7a3f2c/
        bittensor/opentensor-btcli-b8e4d1/
        fastapi/fastapi.tiangolo.com-llmstxt-e1b3c5/
    """
    source_name = _url_to_source_name(url)
    # Sanitize: only allow alphanumeric, dash, dot
    source_name = re.sub(r"[^a-zA-Z0-9.\-]", "-", source_name)

    # Short hash for uniqueness (6 chars)
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:6]

    folder_name = f"{source_name}-{url_hash}"
    return get_tool_cache_dir(tool_id) / folder_name


def get_lance_dir() -> Path:
    """Get path to LanceDB storage directory."""
    return get_toolhub_home() / "lance"


def get_tool_lance_path(tool_id: str) -> Path:
    """Get LanceDB path for a specific tool."""
    return get_lance_dir() / f"{tool_id}.lance"


def get_pid_path() -> Path:
    """Get path to daemon PID file."""
    return get_toolhub_home() / "daemon.pid"


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    home = get_toolhub_home()
    home.mkdir(parents=True, exist_ok=True)
    get_cache_dir().mkdir(exist_ok=True)
    get_lance_dir().mkdir(exist_ok=True)
