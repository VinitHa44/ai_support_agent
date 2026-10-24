"""
RAG (Retrieval-Augmented Generation) Knowledge Base & Hybrid Search for Ampere Docs.
Combines:
1. Dense Vector Embeddings (OpenRouter / OpenAI text-embedding-3-small)
2. Pinecone Vector DB integration (Serverless Vector Index)
3. Sparse Lexical Search (BM25 with token weights)
4. Hybrid Search via Reciprocal Rank Fusion (RRF)
5. Anti-hallucination confidence thresholding
"""

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx

# Ensure project root is in sys.path for direct script and package execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.config import (
        DOCS_PATH,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        PINECONE_API_KEY,
        PINECONE_INDEX_NAME,
    )
    from src.scraper import scrape_all_docs
except ImportError:
    from config import (
        DOCS_PATH,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        PINECONE_API_KEY,
        PINECONE_INDEX_NAME,
    )
    from scraper import scrape_all_docs

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer with stopword filtering."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    stopwords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
        "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
        "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
        "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "shan't", "she", "she'd", "she'll", "she's", "should",
        "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't",
        "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves",
    }
    return [w for w in words if w not in stopwords and len(w) > 1]


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Computes dense vector embedding via OpenRouter or OpenAI API.
    """
    api_key = OPENROUTER_API_KEY.strip() or OPENAI_API_KEY.strip()
    if not api_key:
        return None

    if OPENROUTER_API_KEY.strip():
        base = (OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1").rstrip("/")
        endpoint = f"{base}/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Ampere AI Support Agent",
            "Content-Type": "application/json",
        }
    else:
        base = (OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base}/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY.strip()}",
            "Content-Type": "application/json",
        }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                endpoint,
                headers=headers,
                json={"model": EMBEDDING_MODEL, "input": text[:8000]},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["embedding"]
            else:
                logger.debug(f"Embedding request failed: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        logger.debug(f"Embedding request exception: {e}")
        return None



class DocChunk:
    def __init__(
        self,
        doc_id: str,
        title: str,
        url: str,
        text: str,
        description: str = "",
        section_heading: str = "",
    ):
        self.doc_id = doc_id
        self.title = title
        self.url = url
        self.text = text
        self.description = description
        self.section_heading = section_heading or title
        searchable_text = f"{title} {title} {self.section_heading} {self.section_heading} {description} {description} {text}"
        self.tokens = tokenize(searchable_text)
        self.length = len(self.tokens)


class KnowledgeBase:
    def __init__(self, docs_path: Optional[str] = None):
        self.docs_path = docs_path or DOCS_PATH
        self.chunks: List[DocChunk] = []
        self.chunks_by_id: Dict[str, DocChunk] = {}
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self.pinecone_index = None
        self.load_and_index()
        self._init_pinecone()

    def _init_pinecone(self):
        """Initializes Pinecone Vector DB client if PINECONE_API_KEY is configured."""
        if not PINECONE_API_KEY.strip():
            logger.info("Pinecone API key not configured; RAG search will use BM25.")
            return

        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_API_KEY.strip())
            index_name = PINECONE_INDEX_NAME.strip()

            existing_indexes = []
            for idx in pc.list_indexes():
                name_val = getattr(idx, "name", None) or (idx["name"] if isinstance(idx, dict) else str(idx))
                existing_indexes.append(name_val)
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone serverless index '{index_name}'...")
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self.pinecone_index = pc.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'.")
            self._sync_pinecone_chunks()
        except Exception as e:
            logger.warning(f"Could not initialize Pinecone: {e}.")
            self.pinecone_index = None

    def _sync_pinecone_chunks(self):
        """Upserts documentation chunks directly into Pinecone vector index."""
        if not self.pinecone_index:
            return

        try:
            stats = self.pinecone_index.describe_index_stats()
            total_count = stats.get("total_vector_count", 0)
            if total_count >= len(self.chunks):
                logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already contains {total_count} vectors; ready for queries.")
                return
        except Exception as e:
            logger.debug(f"Could not check Pinecone stats: {e}")

        vectors_to_upsert = []
        for chunk in self.chunks:
            embedding = get_embedding(f"{chunk.title}: {chunk.text[:1000]}")
            if embedding:
                vectors_to_upsert.append({
                    "id": chunk.doc_id,
                    "values": embedding,
                    "metadata": {
                        "title": chunk.title,
                        "heading": chunk.section_heading,
                        "url": chunk.url,
                        "text": chunk.text[:1500],
                        "description": chunk.description,
                    },
                })

        if vectors_to_upsert:
            try:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Upserted {len(vectors_to_upsert)} semantic chunks into Pinecone index.")
            except Exception as e:
                logger.warning(f"Failed to upsert to Pinecone: {e}")

    def _chunk_sections_semantically(
        self,
        sections: List[Dict[str, str]],
        title: str,
        url: str,
        desc: str,
        min_words: int = 50,
        max_words: int = 350,
    ) -> List[Dict[str, str]]:
        """
        Approach 2: Structural / Heading-Aware Semantic Chunking.
        Groups documentation content by section headings into cohesive topic blocks.
        Preserves complete tables, lists, and procedures intact.
        Adds breadcrumb context: [Page Title > Section Heading].
        """
        chunks = []
        curr_heading = title
        curr_parts: List[str] = []
        curr_count = 0

        for sec in sections:
            heading = sec.get("heading", "").strip() or title
            content = sec.get("content", "").strip()
            if not content:
                continue

            words = len(content.split())

            # If section itself is very large (> max_words), split on natural sentence breaks
            if words > max_words:
                if curr_parts:
                    chunk_body = "\n\n".join(curr_parts)
                    chunks.append({
                        "heading": curr_heading,
                        "text": f"[{title} > {curr_heading}]\n{chunk_body}",
                    })
                    curr_parts = []
                    curr_count = 0

                paragraphs = [p.strip() for p in re.split(r"(?<=[.!?])\s+", content) if p.strip()]
                sub_parts: List[str] = []
                sub_count = 0
                for p in paragraphs:
                    p_words = len(p.split())
                    if sub_count >= min_words and (sub_count + p_words > max_words):
                        sub_text = " ".join(sub_parts)
                        chunks.append({
                            "heading": heading,
                            "text": f"[{title} > {heading}]\n{sub_text}",
                        })
                        sub_parts = []
                        sub_count = 0
                    sub_parts.append(p)
                    sub_count += p_words
                if sub_parts:
                    sub_text = " ".join(sub_parts)
                    chunks.append({
                        "heading": heading,
                        "text": f"[{title} > {heading}]\n{sub_text}",
                    })
                continue

            # Boundary condition: if accumulated content is already a complete topic block
            # and a new distinct heading starts with meaningful content
            if curr_count >= min_words and (
                curr_count + words > max_words or (curr_count >= 80 and heading != curr_heading and words >= 40)
            ):
                chunk_body = "\n\n".join(curr_parts)
                chunks.append({
                    "heading": curr_heading,
                    "text": f"[{title} > {curr_heading}]\n{chunk_body}",
                })
                curr_parts = []
                curr_count = 0
                curr_heading = heading

            if not curr_parts:
                curr_heading = heading

            header_prefix = f"### {heading}\n" if heading != title and not content.startswith(heading) else ""
            curr_parts.append(f"{header_prefix}{content}")
            curr_count += words

        if curr_parts:
            chunk_body = "\n\n".join(curr_parts)
            chunks.append({
                "heading": curr_heading,
                "text": f"[{title} > {curr_heading}]\n{chunk_body}",
            })

        return chunks

    def load_and_index(self):
        if not os.path.exists(self.docs_path):
            logger.info(f"Docs not found at {self.docs_path}, scraping now...")
            scrape_all_docs(self.docs_path)

        with open(self.docs_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        self.chunks = []
        self.chunks_by_id = {}
        for i, d in enumerate(raw_docs):
            url = d.get("url", "")
            title = d.get("title", "")
            desc = d.get("description", "")
            content = d.get("content", "")
            sections = d.get("sections")

            if not sections:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
                sections = [{"heading": title, "content": p} for p in paragraphs] or [{"heading": title, "content": content}]

            semantic_chunks = self._chunk_sections_semantically(sections, title, url, desc)
            for k, sc in enumerate(semantic_chunks):
                doc_id = f"doc_{i}_sec_{k}"
                chunk = DocChunk(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    text=sc["text"],
                    description=desc,
                    section_heading=sc.get("heading", title),
                )
                self.chunks.append(chunk)
                self.chunks_by_id[doc_id] = chunk

        # Compute document frequencies for BM25
        self.doc_freq = {}
        total_len = 0
        for chunk in self.chunks:
            total_len += chunk.length
            unique_terms = set(chunk.tokens)
            for term in unique_terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avg_doc_len = total_len / max(len(self.chunks), 1)
        logger.info(f"Indexed {len(self.chunks)} semantic chunks across {len(raw_docs)} documents.")

    def search_bm25(self, query: str, top_k: int = 10, min_score: float = 1.0) -> List[Tuple[DocChunk, float]]:
        """Sparse BM25 Lexical Search."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        n_docs = len(self.chunks)
        scores = []

        for chunk in self.chunks:
            score = 0.0
            term_counts: Dict[str, int] = {}
            for t in chunk.tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt in term_counts:
                    tf = term_counts[qt]
                    df = self.doc_freq.get(qt, 1)
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    denom = tf + self.k1 * (1 - self.b + self.b * (chunk.length / max(self.avg_doc_len, 1)))
                    score += idf * ((tf * (self.k1 + 1)) / denom)

            if score >= min_score:
                scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[DocChunk, float]]:
        """
        Dense Semantic Search using Pinecone Vector Database.
        """
        if not self.pinecone_index:
            return []

        q_vector = get_embedding(query)
        if not q_vector:
            return []

        try:
            results = self.pinecone_index.query(
                vector=q_vector,
                top_k=top_k,
                include_metadata=True,
            )
            pinecone_results = []
            for match in results.get("matches", []):
                chunk_id = match.get("id")
                score = float(match.get("score", 0.0))
                chunk = self.chunks_by_id.get(chunk_id)
                if not chunk and match.get("metadata"):
                    meta = match["metadata"]
                    chunk = DocChunk(
                        doc_id=chunk_id,
                        title=meta.get("title", ""),
                        url=meta.get("url", ""),
                        text=meta.get("text", ""),
                        description=meta.get("description", ""),
                        section_heading=meta.get("heading", ""),
                    )
                if chunk and score >= 0.55:
                    pinecone_results.append((chunk, score))
            return pinecone_results
        except Exception as e:
            logger.warning(f"Pinecone query error: {e}")
            return []

    def search(self, query: str, top_k: int = 3, min_score: float = 1.0) -> List[Tuple[DocChunk, float]]:
        """
        Hybrid Search combining Dense Semantic Search & Sparse BM25 via Reciprocal Rank Fusion (RRF).
        Enforces strict anti-hallucination thresholding.
        """
        sparse_hits = self.search_bm25(query, top_k=10, min_score=min_score)
        dense_hits = self.search_dense(query, top_k=10)

        # If both dense and sparse found nothing, query is out of scope (Anti-Hallucination)
        if not sparse_hits and not dense_hits:
            return []

        # Reciprocal Rank Fusion (RRF)
        rrf_constant = 60
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocChunk] = {}

        for rank, (chunk, _) in enumerate(sparse_hits):
            chunk_map[chunk.doc_id] = chunk
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

        for rank, (chunk, _) in enumerate(dense_hits):
            chunk_map[chunk.doc_id] = chunk
            # Dense matches given slight weight boost in hybrid ranking
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + 1.2 * (1.0 / (rrf_constant + rank + 1))

        combined = [(chunk_map[doc_id], rrf_scores[doc_id]) for doc_id in rrf_scores]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


# Singleton instance
_KB_INSTANCE: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        _KB_INSTANCE = KnowledgeBase()
    return _KB_INSTANCE
