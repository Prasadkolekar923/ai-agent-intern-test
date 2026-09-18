"""BM25/TF-IDF retrieval engine for knowledge base chunks.

Indexes document chunks, calculates lexical match scores, and applies
precedence weighting (boosting active official documents over superseded ones).
"""

import re
from typing import List, Optional, Set, Tuple
from rank_bm25 import BM25Okapi
from app.kb_loader import KBChunk

# Standard English stop words to filter out noise in BM25 scoring
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "at",
    "by", "for", "from", "in", "into", "of", "off", "on", "onto", "over",
    "to", "up", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
    "should", "it", "its", "they", "them", "their", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "my", "your", "our", "what",
    "how", "when", "where", "which", "who", "whom", "why"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms, excluding stop words."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class KBRetriever:
    """Retriever for knowledge base chunks with document precedence ranking."""

    def __init__(
        self,
        chunks: List[KBChunk],
        active_boost: float = 1.3,
        superseded_multiplier: float = 0.95,
        draft_multiplier: float = 0.5,
        default_min_score: float = 1.0,
    ):
        """Initialize retriever with knowledge base chunks and build BM25 index.

        Args:
            chunks: List of KBChunk objects to index.
            active_boost: Multiplier boost for active official documents.
            superseded_multiplier: Multiplier for superseded documents.
            draft_multiplier: Multiplier for draft/unapproved documents.
            default_min_score: Minimum boosted score threshold for returning a match.
        """
        self.chunks = chunks
        self.active_boost = active_boost
        self.superseded_multiplier = superseded_multiplier
        self.draft_multiplier = draft_multiplier
        self.default_min_score = default_min_score

        # Prepare corpus for BM25: emphasize title and heading
        self.tokenized_corpus = []
        for chunk in self.chunks:
            # Emphasize title and heading by repeating in indexed text
            indexed_text = (
                f"{chunk.title} {chunk.title}\n"
                f"{chunk.heading} {chunk.heading}\n"
                f"{chunk.content}"
            )
            tokens = tokenize(indexed_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def _get_precedence_multiplier(self, chunk: KBChunk) -> float:
        """Calculate score multiplier based on chunk status and authority."""
        status = chunk.status.lower()
        authority = chunk.policy_authority.lower()

        if status == "active" and authority == "official":
            return self.active_boost
        elif status == "superseded":
            return self.superseded_multiplier
        elif status == "draft" or authority == "none":
            return self.draft_multiplier
        return 1.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[KBChunk, float]]:
        """Search for top_k relevant chunks given a query.

        Applies precedence weighting favoring active official documents, while
        retaining genuinely relevant superseded documents to allow conflict detection.
        Filters out low-confidence results below min_score.

        Args:
            query: User search query or question.
            top_k: Maximum number of chunks to return.
            min_score: Minimum boosted score threshold. Defaults to self.default_min_score.
                       If set to 0.0 or None, no threshold filtering is performed.

        Returns:
            List of (chunk, score) tuples sorted descending by score.
        """
        if not self.chunks or self.bm25 is None:
            return []

        threshold = self.default_min_score if min_score is None else min_score
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)

        scored_chunks: List[Tuple[KBChunk, float]] = []
        for chunk, raw_score in zip(self.chunks, raw_scores):
            multiplier = self._get_precedence_multiplier(chunk)
            boosted_score = float(raw_score) * multiplier

            if threshold > 0 and boosted_score < threshold:
                continue

            scored_chunks.append((chunk, round(boosted_score, 4)))

        # Sort descending by score
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]

    def is_confident(
        self,
        results: List[Tuple[KBChunk, float]],
        query: Optional[str] = None,
        confidence_threshold: float = 5.0,
        min_coverage: float = 0.35,
    ) -> bool:
        """Check if search results meet a high-confidence threshold.

        Evaluates both the boosted BM25 score of the top result and the
        coverage of non-stop-word query terms in the retrieved document.

        Args:
            results: List of (chunk, score) tuples.
            query: Optional user query string to check term coverage.
            confidence_threshold: Minimum score required for top result to be confident.
            min_coverage: Minimum ratio of query terms that must appear in the top chunk.

        Returns:
            True if the top result meets or exceeds confidence and coverage thresholds.
        """
        if not results:
            return False

        top_chunk, top_score = results[0]
        if top_score < confidence_threshold:
            return False

        if query:
            q_tokens = set(tokenize(query))
            if len(q_tokens) >= 2:
                chunk_tokens = set(
                    tokenize(f"{top_chunk.title} {top_chunk.heading} {top_chunk.content}")
                )
                coverage = len(q_tokens.intersection(chunk_tokens)) / len(q_tokens)
                if coverage < min_coverage:
                    return False

        return True


