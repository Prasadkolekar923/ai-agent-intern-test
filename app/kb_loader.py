"""Knowledge base document parser and chunk extractor.

Parses markdown files in knowledge-base/, extracts front matter metadata,
and chunks content by headings while preserving provenance (filename and heading).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class KBChunk:
    """Represents a chunk of knowledge base document."""
    chunk_id: str
    filename: str
    doc_id: str
    title: str
    heading: str
    content: str
    status: str  # active, superseded, draft, etc.
    policy_authority: str  # official, none, etc.
    metadata: Dict[str, Any]


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML front matter and body from markdown content.

    Args:
        content: Raw markdown text potentially containing YAML front matter.

    Returns:
        A tuple of (metadata_dict, body_text).
    """
    stripped = content.strip()
    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body_text = parts[2].strip()
            parsed = yaml.safe_load(fm_text)
            metadata = parsed if isinstance(parsed, dict) else {}
            return metadata, body_text
    return {}, content.strip()


def _slugify(text: str) -> str:
    """Convert text into a URL-friendly anchor slug."""
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


def load_and_chunk_documents(kb_dir: Path) -> List[KBChunk]:
    """Parse all markdown files in kb_dir and yield structured chunks.

    Chunks documents by headings (H1 intro, H2, H3) while preserving provenance
    metadata such as filename, document ID, title, heading, status, and authority.

    Args:
        kb_dir: Path to directory containing knowledge base markdown files.

    Returns:
        List of structured KBChunk objects.
    """
    if not kb_dir.exists() or not kb_dir.is_dir():
        raise ValueError(f"Knowledge base directory does not exist: {kb_dir}")

    chunks: List[KBChunk] = []
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    for md_file in sorted(kb_dir.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)

        doc_id = str(metadata.get("document_id", md_file.stem))
        title = str(metadata.get("title", md_file.stem))
        status = str(metadata.get("status", "active"))
        authority = str(metadata.get("policy_authority", "official"))
        filename = md_file.name

        matches = list(heading_pattern.finditer(body))
        if not matches:
            # If no markdown headings exist, chunk entire body
            chunks.append(
                KBChunk(
                    chunk_id=f"{filename}#{_slugify(title)}",
                    filename=filename,
                    doc_id=doc_id,
                    title=title,
                    heading=title,
                    content=body,
                    status=status,
                    policy_authority=authority,
                    metadata=metadata,
                )
            )
            continue

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_content = body[start:end].strip()

            # For H1 (doc title), only create a chunk if there is text before the first H2/H3
            if level == 1:
                if section_content:
                    chunks.append(
                        KBChunk(
                            chunk_id=f"{filename}#{_slugify(heading_text)}",
                            filename=filename,
                            doc_id=doc_id,
                            title=title,
                            heading=heading_text,
                            content=section_content,
                            status=status,
                            policy_authority=authority,
                            metadata=metadata,
                        )
                    )
            else:
                chunks.append(
                    KBChunk(
                        chunk_id=f"{filename}#{_slugify(heading_text)}",
                        filename=filename,
                        doc_id=doc_id,
                        title=title,
                        heading=heading_text,
                        content=section_content,
                        status=status,
                        policy_authority=authority,
                        metadata=metadata,
                    )
                )

    return chunks

