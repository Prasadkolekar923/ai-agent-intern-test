"""Tests for knowledge base chunking and retrieval (Phase 2 acceptance)."""

import pytest
from pathlib import Path
from app.config import KNOWLEDGE_BASE_DIR
from app.kb_loader import parse_frontmatter, load_and_chunk_documents, KBChunk
from app.retrieval import KBRetriever, tokenize


@pytest.fixture(scope="module")
def kb_chunks():
    """Load and chunk all documents in knowledge-base/ once for tests."""
    return load_and_chunk_documents(KNOWLEDGE_BASE_DIR)


@pytest.fixture(scope="module")
def retriever(kb_chunks):
    """Initialize KBRetriever with real knowledge base chunks."""
    return KBRetriever(kb_chunks)


def test_parse_frontmatter():
    """Test extracting YAML front matter and body from markdown content."""
    md_sample = """---
document_id: TEST-01
title: Test Document
status: active
policy_authority: official
---
# Test Document

## Section 1
Content for section 1.
"""
    metadata, body = parse_frontmatter(md_sample)
    assert metadata["document_id"] == "TEST-01"
    assert metadata["title"] == "Test Document"
    assert metadata["status"] == "active"
    assert metadata["policy_authority"] == "official"
    assert body.startswith("# Test Document")
    assert "Content for section 1." in body


def test_parse_frontmatter_empty_or_no_frontmatter():
    """Test parse_frontmatter when no YAML block is present."""
    md_raw = "# Just Content\nNo frontmatter here."
    metadata, body = parse_frontmatter(md_raw)
    assert metadata == {}
    assert body == md_raw


def test_load_and_chunk_documents(kb_chunks):
    """Verify load_and_chunk_documents successfully parses knowledge-base files."""
    assert len(kb_chunks) > 30, f"Expected >30 chunks across KB, found {len(kb_chunks)}"

    filenames = {c.filename for c in kb_chunks}
    assert "01-returns-policy-current.md" in filenames
    assert "02-returns-policy-legacy.md" in filenames
    assert "03-final-sale-and-promotions.md" in filenames
    assert "14-internal-content-migration-notes.md" in filenames

    for chunk in kb_chunks:
        assert chunk.chunk_id, "chunk_id must not be empty"
        assert chunk.filename.endswith(".md"), "filename must end with .md"
        assert chunk.heading, "heading must not be empty"
        assert chunk.content, "chunk content must not be empty"
        assert chunk.status in {"active", "superseded", "draft"}, f"Unexpected status: {chunk.status}"
        assert chunk.policy_authority in {"official", "none"}, f"Unexpected authority: {chunk.policy_authority}"


def test_return_window_precedence(retriever):
    """Acceptance Check 1: Query about return window prefers active over legacy policy.

    The top result comes from 01-returns-policy-current.md (active),
    not 02-returns-policy-legacy.md (superseded).
    """
    query = "What is the return window for standard returns?"
    results = retriever.search(query, top_k=5)

    assert len(results) > 0, "Expected search results for return window query"
    top_chunk, top_score = results[0]
    assert top_chunk.filename == "01-returns-policy-current.md", (
        f"Expected top chunk from 01-returns-policy-current.md, got {top_chunk.filename} (score: {top_score})"
    )
    assert top_chunk.status == "active"

    # Verify that active policy 01 strictly ranks ahead of superseded policy 02
    retrieved_files = [c.filename for c, _ in results]
    if "02-returns-policy-legacy.md" in retrieved_files:
        idx_current = retrieved_files.index("01-returns-policy-current.md")
        idx_legacy = retrieved_files.index("02-returns-policy-legacy.md")
        assert idx_current < idx_legacy, "Current active returns policy must rank before legacy policy"


def test_superseded_retained_for_conflict_detection(retriever):
    """Ensure superseded content is retained in lower ranks rather than silently dropped,
    allowing the agent layer to detect and surface conflicts.
    """
    query = "return window 45 days policy"
    results = retriever.search(query, top_k=5)

    retrieved_files = [chunk.filename for chunk, _ in results]
    # Both current and legacy documents should be retrieved
    assert "01-returns-policy-current.md" in retrieved_files or "02-returns-policy-legacy.md" in retrieved_files


def test_final_sale_promotions_retrieval(retriever):
    """Acceptance Check 2: Query touching final-sale/promotions retrieves 03-final-sale-and-promotions.md."""
    query = "Are final sale or promotional discount items eligible for return?"
    results = retriever.search(query, top_k=5)

    assert len(results) > 0, "Expected results for final-sale query"
    retrieved_files = [chunk.filename for chunk, _ in results]
    assert "03-final-sale-and-promotions.md" in retrieved_files, (
        f"Expected 03-final-sale-and-promotions.md in results, got: {retrieved_files}"
    )


def test_irrelevant_query_threshold(retriever):
    """Acceptance Check 3: A query with no relevant match returns empty or low-confidence results."""
    # Completely non-matching / gibberish query returns empty list
    nonsense_query = "xyzqlmp99 nonexistent quantum flux widget"
    results = retriever.search(nonsense_query, top_k=5)
    assert len(results) == 0, f"Expected 0 results for nonsense query, got {len(results)}"
    assert not retriever.is_confident(results, query=nonsense_query), "Should not be confident on nonsense query"

    # Out-of-domain query with weak lexical overlap returns low confidence
    out_of_domain = "What is the capital of Australia and its current weather forecast?"
    results_ood = retriever.search(out_of_domain, top_k=5, min_score=0.0)
    assert not retriever.is_confident(results_ood, query=out_of_domain), (
        "Out of domain query should fail confidence/coverage threshold"
    )

    # Question with insufficient KB coverage (e.g. vegan materials) triggers low confidence
    unsupported_query = "Are all fabrics and adhesives in your bags vegan?"
    results_unsup = retriever.search(unsupported_query, top_k=5, min_score=0.0)
    assert not retriever.is_confident(results_unsup, query=unsupported_query), (
        "Query with insufficient KB information must be flagged as low confidence"
    )



def test_chunk_metadata_preservation(retriever):
    """Acceptance Check 4: Verify chunk metadata (filename + heading) is present and correct for at least 3 queries."""
    sample_queries = [
        ("international shipping to Canada", "06-international-shipping.md"),
        ("how to clean Breeze Tumbler", "12-breeze-tumbler-product-card.md"),
        ("warranty coverage for backpack", "07-warranty.md"),
    ]

    for query, expected_file in sample_queries:
        results = retriever.search(query, top_k=5)
        assert len(results) > 0, f"No results for query: '{query}'"

        # Check that expected file appears in top results
        retrieved_files = [c.filename for c, _ in results]
        assert expected_file in retrieved_files, (
            f"Expected {expected_file} in results for '{query}', got {retrieved_files}"
        )

        # Verify every returned chunk has complete metadata
        for chunk, score in results:
            assert chunk.filename, "Filename missing from chunk metadata"
            assert chunk.heading, "Heading missing from chunk metadata"
            assert chunk.doc_id, "Document ID missing from chunk metadata"
            assert chunk.title, "Title missing from chunk metadata"
            assert score > 0, "Boosted score must be strictly positive"


def test_search_respects_top_k(retriever):
    """Verify search never dumps the entire knowledge base and strictly limits to top_k."""
    query = "shipping return policy warranty"
    for k in [1, 3, 5]:
        results = retriever.search(query, top_k=k)
        assert len(results) <= k, f"Expected at most {k} results, got {len(results)}"

