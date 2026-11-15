"""Regression tests for heading-aware Markdown chunks and weighted RRF."""
import pytest

from rag.chunking import chunk_markdown
from rag.index import build_passages
from rag.retrieval import Passage, weighted_rrf, HybridRetriever


def test_markdown_preserves_titles_and_word_boundaries():
    markdown = (
        "# Coding Agent\n\n"
        "## Verification\n\n"
        "The successful repair requires a fresh test result after editing.\n\n"
        "Fresh tests must use the current edit version.\n\n"
        "## Recovery\n\n"
        "Repeated failures should stop the agent instead of retrying forever."
    )
    chunks = chunk_markdown(markdown, size=115, overlap=12)
    assert chunks
    assert all(len(c) <= 115 for c in chunks)
    assert any("Coding Agent / Verification" in c for c in chunks)
    assert any("Coding Agent / Recovery" in c for c in chunks)
    assert any("successful repair" in c for c in chunks)
    assert all(not c.startswith("ul repair") for c in chunks)


def test_markdown_headings_in_code_fence_are_not_section_titles():
    markdown = "# Guide\n\n\x60\x60\x60python\n# not a header\nprint(1)\n\x60\x60\x60\n"
    chunks = chunk_markdown(markdown, size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].startswith("Guide\n")
    assert "# not a header" in chunks[0]


def test_source_ids_remain_citable(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Rules\n\n## Verify\n\nAlways run the test.", encoding="utf-8"
    )
    passages = build_passages(tmp_path, chunk_size=100, overlap=10)
    assert passages[0].source_id == "guide.md#chunk-0"
    assert "Rules / Verify" in passages[0].text


def test_weighted_rrf_ranking_and_default_equivalence():
    p1, p2 = Passage("a#chunk-0", "a", "A"), Passage("b#chunk-0", "b", "B")
    sparse = [(p1, 2.0), (p2, 1.0)]
    dense = [(p2, 0.9), (p1, 0.8)]
    equal = weighted_rrf([sparse, dense])
    assert equal[0][0].source_id == "a#chunk-0"  # deterministic tie break
    weighted = weighted_rrf([sparse, dense], weights=(2, 1))
    assert weighted[0][0].source_id == "a#chunk-0"
    weighted_dense = weighted_rrf([sparse, dense], weights=(1, 2))
    assert weighted_dense[0][0].source_id == "b#chunk-0"


@pytest.mark.parametrize("weights", [(0, 0), (-1, 1), (float("nan"), 1)])
def test_invalid_rrf_weights_rejected(weights):
    with pytest.raises(ValueError):
        HybridRetriever([], bm25_weight=weights[0], dense_weight=weights[1])
