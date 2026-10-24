"""
Tests for Document Search, RAG retrieval, citations, and unknown fallback.
"""

from src.rag import get_knowledge_base
from src.tools import execute_search_docs


def test_search_add_agent():
    res = execute_search_docs("How do I add an agent?")
    assert res["found"] is True
    assert len(res["results"]) > 0
    assert len(res["doc_links"]) > 0
    # Must contain official Ampere doc link
    assert any("docs.ampere.sh" in link for link in res["doc_links"])


def test_search_credits_and_pricing():
    res = execute_search_docs("How do credits and pricing work in Ampere?")
    assert res["found"] is True
    assert any("credits-and-pricing" in link for link in res["doc_links"])


def test_unrelated_query_no_hallucination():
    res = execute_search_docs("How do I bake a chocolate cake?")
    assert res["found"] is False
    assert len(res["results"]) == 0
