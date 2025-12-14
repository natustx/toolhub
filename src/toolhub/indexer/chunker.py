"""Markdown chunker for splitting documents into indexable chunks.

Splits markdown by headers and code blocks, targeting ~500 tokens per chunk.
Each chunk includes metadata about its source file and heading hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A chunk of text with metadata for indexing."""

    content: str
    source_file: str
    headings: list[str] = field(default_factory=list)
    is_code: bool = False

    @property
    def heading(self) -> str | None:
        """Return the most specific (deepest) heading."""
        return self.headings[-1] if self.headings else None

    @property
    def heading_path(self) -> str:
        """Return full heading path like 'Installation > Configuration > Options'."""
        return " > ".join(self.headings) if self.headings else ""

    def estimated_tokens(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4


# Regex patterns
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return len(text) // 4


def _split_by_headers(content: str) -> list[tuple[list[str], str]]:
    """Split content by headers, tracking heading hierarchy.

    Returns list of (heading_stack, section_content) tuples.
    """
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_content: list[str] = []
    current_headings: list[str] = []

    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        header_match = HEADER_PATTERN.match(line)

        if header_match:
            # Save previous section if it has content
            section_text = "\n".join(current_content).strip()
            if section_text:
                sections.append((list(current_headings), section_text))

            # Update heading stack
            level = len(header_match.group(1))
            heading_text = header_match.group(2).strip()

            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading_text))
            current_headings = [h[1] for h in heading_stack]
            current_content = []
        else:
            current_content.append(line)

        i += 1

    # Don't forget the last section
    section_text = "\n".join(current_content).strip()
    if section_text:
        sections.append((list(current_headings), section_text))

    return sections


def _split_long_section(
    content: str, headings: list[str], max_tokens: int
) -> list[tuple[list[str], str]]:
    """Split a long section into smaller chunks.

    Tries to split on paragraph boundaries, then sentences, then hard split.
    """
    if _estimate_tokens(content) <= max_tokens:
        return [(headings, content)]

    chunks: list[tuple[list[str], str]] = []

    # Try splitting on double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", content)
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append((headings, "\n\n".join(current_chunk)))
            current_chunk = []
            current_tokens = 0

        # If single paragraph is too long, split it further
        if para_tokens > max_tokens:
            # Split on sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sent_tokens = _estimate_tokens(sentence)
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunks.append((headings, "\n\n".join(current_chunk)))
                    current_chunk = []
                    current_tokens = 0

                # Hard split if single sentence is too long
                if sent_tokens > max_tokens:
                    # Split at max_tokens * 4 characters
                    max_chars = max_tokens * 4
                    for j in range(0, len(sentence), max_chars):
                        chunk_text = sentence[j : j + max_chars]
                        chunks.append((headings, chunk_text))
                else:
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append((headings, "\n\n".join(current_chunk)))

    return chunks


def _extract_code_blocks(content: str) -> list[tuple[str, bool]]:
    """Extract code blocks and text separately.

    Returns list of (content, is_code) tuples.
    """
    parts: list[tuple[str, bool]] = []
    last_end = 0

    for match in CODE_BLOCK_PATTERN.finditer(content):
        # Text before code block
        text_before = content[last_end : match.start()].strip()
        if text_before:
            parts.append((text_before, False))

        # Code block itself (include the full match with backticks for context)
        parts.append((match.group(0), True))
        last_end = match.end()

    # Remaining text after last code block
    remaining = content[last_end:].strip()
    if remaining:
        parts.append((remaining, False))

    return parts if parts else [(content, False)]


def chunk_markdown(
    content: str,
    source_file: str,
    max_tokens: int = 500,
) -> list[Chunk]:
    """Chunk markdown content into indexable pieces.

    Args:
        content: Markdown text to chunk
        source_file: Path/name of source file for metadata
        max_tokens: Target maximum tokens per chunk (approximate)

    Returns:
        List of Chunk objects ready for embedding
    """
    chunks: list[Chunk] = []

    # First split by headers
    sections = _split_by_headers(content)

    for headings, section_content in sections:
        # Extract code blocks separately
        parts = _extract_code_blocks(section_content)

        for part_content, is_code in parts:
            if not part_content.strip():
                continue

            # Split long sections
            if _estimate_tokens(part_content) > max_tokens and not is_code:
                sub_chunks = _split_long_section(part_content, headings, max_tokens)
                for sub_headings, sub_content in sub_chunks:
                    chunks.append(
                        Chunk(
                            content=sub_content,
                            source_file=source_file,
                            headings=sub_headings,
                            is_code=False,
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        content=part_content,
                        source_file=source_file,
                        headings=headings,
                        is_code=is_code,
                    )
                )

    return chunks


def chunk_file(file_path: Path, max_tokens: int = 500) -> list[Chunk]:
    """Chunk a markdown file.

    Args:
        file_path: Path to markdown file
        max_tokens: Target maximum tokens per chunk

    Returns:
        List of Chunk objects
    """
    content = file_path.read_text(encoding="utf-8")
    return chunk_markdown(content, source_file=str(file_path), max_tokens=max_tokens)


def chunk_directory(
    dir_path: Path,
    max_tokens: int = 500,
    extensions: tuple[str, ...] = (".md", ".markdown", ".rst", ".txt"),
) -> list[Chunk]:
    """Chunk all documentation files in a directory.

    Args:
        dir_path: Directory to process
        max_tokens: Target maximum tokens per chunk
        extensions: File extensions to include

    Returns:
        List of Chunk objects from all files
    """
    chunks: list[Chunk] = []

    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                file_chunks = chunk_file(file_path, max_tokens=max_tokens)
                # Use relative path for cleaner source references
                rel_path = file_path.relative_to(dir_path)
                for chunk in file_chunks:
                    chunk.source_file = str(rel_path)
                chunks.extend(file_chunks)
            except Exception:
                # Skip files that can't be read
                continue

    return chunks
