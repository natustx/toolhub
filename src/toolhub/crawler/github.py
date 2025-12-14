"""GitHub repository crawler for documentation extraction."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import git

from toolhub.crawler.base import Crawler, CrawlResult

logger = logging.getLogger(__name__)

# Patterns for documentation files
DOC_PATTERNS = [
    "README*",
    "readme*",
    "docs/**/*",
    "doc/**/*",
    "documentation/**/*",
    "examples/**/*",
    "example/**/*",
    "*.md",
    "*.rst",
    "*.txt",
]

# Extensions to include
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".markdown"}

# Extensions for API specs (always included if they match patterns)
SPEC_EXTENSIONS = {".json", ".yaml", ".yml"}

# Files/dirs to exclude
EXCLUDE_PATTERNS = {
    "node_modules",
    ".git",
    ".github",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


def parse_github_url(url: str) -> tuple[str, str, str | None]:
    """Parse a GitHub URL into owner, repo, and optional ref.

    Args:
        url: GitHub URL (https://github.com/owner/repo or git@github.com:owner/repo)

    Returns:
        Tuple of (owner, repo, ref) where ref may be None

    Raises:
        ValueError: If URL cannot be parsed
    """
    # Handle git@ URLs
    if url.startswith("git@"):
        match = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
        if match:
            return match.group(1), match.group(2), None
        raise ValueError(f"Invalid GitHub SSH URL: {url}")

    # Handle https URLs
    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise ValueError(f"Not a GitHub URL: {url}")

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL path: {url}")

    owner = path_parts[0]
    repo = path_parts[1].removesuffix(".git")

    # Check for tree/blob refs
    ref = None
    if len(path_parts) >= 4 and path_parts[2] in ("tree", "blob"):
        ref = path_parts[3]

    return owner, repo, ref


def is_openapi_file(path: Path) -> bool:
    """Check if a file looks like an OpenAPI/Swagger spec."""
    name_lower = path.name.lower()
    if path.suffix.lower() not in SPEC_EXTENSIONS:
        return False
    return any(x in name_lower for x in ["openapi", "swagger", "api.json", "api.yaml", "api.yml"])


def should_include_file(path: Path) -> bool:
    """Check if a file should be included in the crawl."""
    # Always include OpenAPI specs
    if is_openapi_file(path):
        return True

    # Check extension for docs
    if path.suffix.lower() not in DOC_EXTENSIONS:
        return False

    # Check against exclude patterns
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return False
        for pattern in EXCLUDE_PATTERNS:
            if pattern.startswith("*") and part.endswith(pattern[1:]):
                return False

    return True


class GitHubCrawler(Crawler):
    """Crawler for GitHub repositories.

    Performs shallow clone and extracts documentation files.
    """

    def __init__(self, github_token: str | None = None):
        """Initialize the GitHub crawler.

        Args:
            github_token: Optional GitHub token for private repos
        """
        self.github_token = github_token

    @property
    def source_type(self) -> str:
        return "github"

    def can_handle(self, url: str) -> bool:
        """Check if URL is a GitHub repository."""
        try:
            parse_github_url(url)
            return True
        except ValueError:
            return False

    def _get_clone_url(self, url: str) -> str:
        """Get the clone URL, adding auth token if available."""
        owner, repo, _ = parse_github_url(url)
        base_url = f"https://github.com/{owner}/{repo}.git"

        if self.github_token:
            return f"https://{self.github_token}@github.com/{owner}/{repo}.git"
        return base_url

    def crawl(self, url: str, cache_dir: Path) -> CrawlResult:
        """Clone a GitHub repo and extract documentation files.

        Args:
            url: GitHub repository URL
            cache_dir: Directory to save extracted documentation

        Returns:
            CrawlResult with status and list of files
        """
        try:
            owner, repo, ref = parse_github_url(url)
        except ValueError as e:
            return CrawlResult(
                source_url=url,
                source_type=self.source_type,
                cache_dir=cache_dir,
                files_crawled=[],
                error=str(e),
            )

        clone_url = self._get_clone_url(url)

        # Create temp dir for clone
        temp_dir = Path(tempfile.mkdtemp(prefix="toolhub_"))

        try:
            # Shallow clone
            logger.info(f"Cloning {owner}/{repo}...")
            clone_args = {"depth": 1}
            if ref:
                clone_args["branch"] = ref

            try:
                git.Repo.clone_from(clone_url, temp_dir, **clone_args)
            except git.GitCommandError as e:
                return CrawlResult(
                    source_url=url,
                    source_type=self.source_type,
                    cache_dir=cache_dir,
                    files_crawled=[],
                    error=f"Git clone failed: {e}",
                )

            # Ensure cache dir exists and is clean
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Find and copy documentation files
            files_crawled = []
            for file_path in temp_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                if not should_include_file(file_path):
                    continue

                # Determine destination path
                rel_path = file_path.relative_to(temp_dir)
                dest_path = cache_dir / rel_path

                # Create parent dirs and copy
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                files_crawled.append(dest_path)
                logger.debug(f"Copied: {rel_path}")

            logger.info(f"Extracted {len(files_crawled)} documentation files")

            return CrawlResult(
                source_url=url,
                source_type=self.source_type,
                cache_dir=cache_dir,
                files_crawled=files_crawled,
            )

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
