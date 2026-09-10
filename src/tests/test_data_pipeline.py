"""
Data Pipeline Smoke Tests
=========================
Tests ``preprocess``, ``graph_builder``, and ``dataset`` using tiny dummy
data — no real PolitiFact files or internet access required.

Run with::

    python -m pytest src/tests/test_data_pipeline.py -v
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch


# ===========================================================================
# preprocess.py
# ===========================================================================

class TestCleanText:
    """Tests for :func:`src.data.preprocess.clean_text`."""

    def test_none_returns_empty(self):
        from src.data.preprocess import clean_text
        assert clean_text(None) == ""

    def test_nan_returns_empty(self):
        import math
        from src.data.preprocess import clean_text
        assert clean_text(float("nan")) == ""

    def test_strips_html_tags(self):
        from src.data.preprocess import clean_text
        assert clean_text("<b>Hello</b>") == "Hello"

    def test_decodes_html_entities(self):
        from src.data.preprocess import clean_text
        assert "&" in clean_text("Cats &amp; Dogs")

    def test_collapses_whitespace(self):
        from src.data.preprocess import clean_text
        assert clean_text("Hello   \n\t  World") == "Hello World"

    def test_strips_leading_trailing(self):
        from src.data.preprocess import clean_text
        assert clean_text("  hello  ") == "hello"

    def test_plain_string_unchanged(self):
        from src.data.preprocess import clean_text
        assert clean_text("Obama signs bill") == "Obama signs bill"


# ===========================================================================
# graph_builder.py
# ===========================================================================

class TestGraphBuilder:
    """Tests for :class:`src.data.graph_builder.GraphBuilder`.

    Uses a monkeypatched ``_embed`` method so sentence-transformers is not
    required at test time.
    """

    EMB_DIM = 8  # small embedding dim for tests

    @pytest.fixture
    def builder_with_mock(self, monkeypatch):
        """Return a GraphBuilder whose _embed is replaced with a deterministic mock."""
        from src.data.graph_builder import GraphBuilder

        builder = GraphBuilder()

        def _mock_embed(texts: list[str]) -> torch.Tensor:
            n = len(texts)
            x = torch.randn(n, self.EMB_DIM)
            return x / x.norm(dim=1, keepdim=True)

        monkeypatch.setattr(builder, "_embed", _mock_embed)
        return builder

    def test_build_with_explicit_hashtag(self, builder_with_mock):
        graph = builder_with_mock.build("Obama signs bill #Healthcare #2024")
        assert "news" in graph.node_types
        assert "hashtag" in graph.node_types
        assert graph["news"].x.shape == (1, self.EMB_DIM)
        assert graph["hashtag"].x.shape[1] == self.EMB_DIM
        assert graph["hashtag"].x.shape[0] >= 1  # at least 1 hashtag
        ei = graph["news", "has_hashtag", "hashtag"].edge_index
        assert ei.shape[0] == 2
        assert ei.shape[1] == graph["hashtag"].x.shape[0]

    def test_build_with_no_hashtags_no_error(self, builder_with_mock):
        # No # tags, but title is short lowercase — should produce empty hashtag nodes
        graph = builder_with_mock.build("a simple claim with no tags")
        assert "news" in graph.node_types
        assert graph["news"].x.shape == (1, self.EMB_DIM)
        # edge_index should exist even if empty
        ei = graph["news", "has_hashtag", "hashtag"].edge_index
        assert ei.shape[0] == 2

    def test_reverse_edges_present(self, builder_with_mock):
        graph = builder_with_mock.build("Claim #FakeNews")
        assert ("hashtag", "rev_has_hashtag", "news") in graph.edge_types

    def test_metadata_returns_correct_types(self):
        from src.data.graph_builder import GraphBuilder
        builder = GraphBuilder()
        node_types, edge_types = builder.metadata()
        assert "news" in node_types
        assert "hashtag" in node_types
        assert ("news", "has_hashtag", "hashtag") in edge_types
        assert ("hashtag", "rev_has_hashtag", "news") in edge_types


# ===========================================================================
# dataset.py — load_fakenewsnet + stratified_split + Dataset + collate_fn
# ===========================================================================

@pytest.fixture
def dummy_raw_dir(tmp_path):
    """Create tiny fake FakeNewsNet CSVs in a temp directory."""
    fake_rows = [
        {"id": f"fake_{i}", "url": "http://x.com", "title": f"Fake claim {i}", "tweet_ids": ""}
        for i in range(6)
    ]
    real_rows = [
        {"id": f"real_{i}", "url": "http://y.com", "title": f"Real claim {i}", "tweet_ids": ""}
        for i in range(6)
    ]

    for filename, rows in [
        ("politifact_fake.csv", fake_rows),
        ("politifact_real.csv", real_rows),
    ]:
        path = tmp_path / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "url", "title", "tweet_ids"])
            writer.writeheader()
            writer.writerows(rows)

    return tmp_path


class TestLoadFakeNewsNet:
    def test_returns_correct_count(self, dummy_raw_dir):
        from src.data.dataset import load_fakenewsnet
        records = load_fakenewsnet(dummy_raw_dir)
        assert len(records) == 12  # 6 fake + 6 real

    def test_labels_correct(self, dummy_raw_dir):
        from src.data.dataset import load_fakenewsnet, LABEL_FAKE, LABEL_REAL
        records = load_fakenewsnet(dummy_raw_dir)
        labels = {r["label"] for r in records}
        assert labels == {LABEL_FAKE, LABEL_REAL}

    def test_missing_csv_raises(self, tmp_path):
        from src.data.dataset import load_fakenewsnet
        with pytest.raises(FileNotFoundError):
            load_fakenewsnet(tmp_path)  # empty dir → no CSVs


class TestStratifiedSplit:
    def _make_samples(self, n_fake=10, n_real=10):
        from src.data.dataset import PolitiFactSample, LABEL_FAKE, LABEL_REAL
        import torch
        from torch_geometric.data import HeteroData

        dummy_tokens = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

        def _make_graph():
            g = HeteroData()
            g["news"].x = torch.randn(1, 4)
            g["hashtag"].x = torch.zeros(0, 4)
            g["news", "has_hashtag", "hashtag"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            g["hashtag", "rev_has_hashtag", "news"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            return g

        samples = []
        for i in range(n_fake):
            samples.append(PolitiFactSample(
                id=f"fake_{i}", title=f"fake {i}", label=LABEL_FAKE,
                tokens=dummy_tokens, graph=_make_graph(),
            ))
        for i in range(n_real):
            samples.append(PolitiFactSample(
                id=f"real_{i}", title=f"real {i}", label=LABEL_REAL,
                tokens=dummy_tokens, graph=_make_graph(),
            ))
        return samples

    def test_split_sizes_sum_to_total(self):
        from src.data.dataset import stratified_split
        samples = self._make_samples(20, 20)
        train, val, test = stratified_split(samples, [0.7, 0.1, 0.2], seed=42)
        assert len(train) + len(val) + len(test) == len(samples)

    def test_no_overlap_between_splits(self):
        from src.data.dataset import stratified_split
        samples = self._make_samples(20, 20)
        train, val, test = stratified_split(samples, [0.7, 0.1, 0.2], seed=42)
        train_ids = {s.id for s in train}
        val_ids   = {s.id for s in val}
        test_ids  = {s.id for s in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_both_labels_in_train(self):
        from src.data.dataset import stratified_split, LABEL_FAKE, LABEL_REAL
        samples = self._make_samples(20, 20)
        train, _, _ = stratified_split(samples, [0.7, 0.1, 0.2], seed=42)
        train_labels = {s.label for s in train}
        assert LABEL_FAKE in train_labels and LABEL_REAL in train_labels


class TestCollateFn:
    def _make_batch(self, size=4):
        from src.data.dataset import PolitiFactSample
        import torch
        from torch_geometric.data import HeteroData

        batch = []
        for i in range(size):
            tokens = {
                "input_ids":      torch.randint(0, 100, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }
            g = HeteroData()
            g["news"].x = torch.randn(1, 4)
            g["hashtag"].x = torch.zeros(0, 4)
            g["news", "has_hashtag", "hashtag"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            g["hashtag", "rev_has_hashtag", "news"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            batch.append(PolitiFactSample(
                id=str(i), title=f"claim {i}", label=i % 2,
                tokens=tokens, graph=g,
            ))
        return batch

    def test_output_keys(self):
        from src.data.dataset import collate_fn
        out = collate_fn(self._make_batch(4))
        assert set(out.keys()) == {"input_ids", "attention_mask", "labels", "graph_batch"}

    def test_input_ids_shape(self):
        from src.data.dataset import collate_fn
        out = collate_fn(self._make_batch(4))
        assert out["input_ids"].shape == (4, 8)

    def test_labels_shape(self):
        from src.data.dataset import collate_fn
        out = collate_fn(self._make_batch(4))
        assert out["labels"].shape == (4,)
        assert out["labels"].dtype == torch.long

    def test_graph_batch_num_graphs(self):
        from src.data.dataset import collate_fn
        out = collate_fn(self._make_batch(4))
        assert out["graph_batch"].num_graphs == 4
