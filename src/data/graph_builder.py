"""
Heterogeneous Graph Construction (News + Hashtag)
==================================================
Builds a **heterogeneous** ``torch_geometric.data.HeteroData`` graph that
partially reproduces the M3DUSA schema (Martirano et al., 2025).

Full M3DUSA schema (requires Twitter API)::

    Nodes: News · Tweet · User · Hashtag
    Edges: discusses · has_hashtag · posted · retweeted · mentions

This module implements the **API-free subset** (Option B):

    Nodes: News (1) · Hashtag (H)
    Edges: (News, has_hashtag, Hashtag)  — one edge per hashtag in the claim

Hashtags are extracted from the claim title:

  * Explicit ``#tag`` tokens in the text.
  * Capitalised multi-word phrases (TitleCase runs) heuristically treated as
    implicit hashtag candidates when no explicit ``#`` tags are present.

Node features
-------------
- ``news`` nodes: sentence embedding of the full claim title ``(1, D)``
- ``hashtag`` nodes: sentence embedding of each hashtag string ``(H, D)``

When you later obtain Twitter API access, extend this class by overriding
:meth:`GraphBuilder.build` to accept hydrated tweet/user/hashtag dicts and
return a full four-type ``HeteroData`` object.

Usage::

    from src.data.graph_builder import GraphBuilder

    builder = GraphBuilder()

    graph = builder.build("Trump claims election was stolen #FakeNews #2020")
    print(type(graph))                              # HeteroData
    print(graph["news"].x.shape)                    # (1, 384)
    print(graph["hashtag"].x.shape)                 # (H, 384)
    print(graph["news", "has_hashtag", "hashtag"].edge_index.shape)  # (2, H)

Reference
---------
Martirano, L., Comito, C., Guarascio, M., Pisani, F.S. & Zicari, P. (2025).
M3DUSA: A Modular Multi-Modal Deep fUSion Architecture for fake news detection
on social media. SNAM. https://github.com/lilymart/M3DUSA
"""
from __future__ import annotations

import re
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from src.utils.logging import get_logger

log = get_logger(__name__)

# Default sentence encoder — fast, 384-dim, CPU-friendly
_DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum length (chars) for a hashtag candidate to be accepted
_MIN_HASHTAG_LEN = 3

# Common English words that appear title-cased but are not proper nouns.
# Used to filter the implicit hashtag fallback.
_STOPWORDS: frozenset[str] = frozenset({
    # Articles / determiners
    "the", "this", "that", "these", "those", "their", "there",
    # Prepositions
    "about", "after", "against", "amid", "among", "around", "before",
    "behind", "below", "beside", "between", "beyond", "despite", "down",
    "during", "except", "from", "into", "like", "near", "onto", "over",
    "past", "since", "through", "till", "toward", "under", "until",
    "upon", "with", "within", "without",
    # Conjunctions / connectives
    "also", "although", "because", "both", "either", "even", "however",
    "just", "neither", "once", "only", "rather", "since", "still",
    "then", "though", "unless", "until", "when", "where", "whether",
    "while", "whom", "whose",
    # Common verbs / auxiliaries used in titles
    "been", "being", "claim", "claims", "could", "declares", "does",
    "gets", "give", "given", "goes", "have", "having", "here", "know",
    "make", "makes", "meet", "more", "need", "never", "orders", "paid",
    "pays", "puts", "says", "seem", "send", "show", "shows", "sign",
    "signs", "take", "talk", "tells", "that", "them", "they", "took",
    "used", "uses", "want", "ways", "will", "work", "works",
    # Common title-case starters / sentence words
    "breaking", "court", "first", "here", "just", "last", "most",
    "much", "must", "next", "news", "only", "open", "part", "pays",
    "plan", "plus", "poll", "says", "soon", "such", "than", "then",
    "time", "told", "turn", "very", "well", "went", "what", "will",
    "with", "year", "your",
    # Numbers / ordinals as words
    "second", "third", "fourth", "fifth",
})


# ---------------------------------------------------------------------------
# Hashtag extraction helpers
# ---------------------------------------------------------------------------

def extract_hashtags(text: str) -> list[str]:
    """Extract semantically meaningful hashtag strings from *text*.

    Two strategies (applied in order):

    1. **Explicit** — ``#word`` tokens in the text: ``#FakeNews`` → ``fakenews``.
       All non-empty explicit tags (≥ ``_MIN_HASHTAG_LEN`` chars) are kept as-is.

    2. **Implicit fallback** — used only when no explicit ``#`` tags are found.
       Extracts **proper nouns and acronyms** only:

       - **Acronyms**: ALL-CAPS tokens ≥ 2 chars (e.g. ``NFL``, ``FBI``, ``POTUS``)
       - **Proper nouns**: Title-case tokens ≥ 4 chars that are *not* in the
         stopword list (e.g. ``Obama``, ``Trump``, ``Congress``).

       Common function words (``That``, ``Over``, ``First``, ``This``, …) and
       sentence-initial capitalised words are filtered out by the stopword set,
       leaving only genuinely named entities as graph nodes.

    Args:
        text: Claim title string (already cleaned).

    Returns:
        Deduplicated list of lowercase hashtag strings.  Empty list if none found.
    """
    # 1. Explicit #hashtags — always preferred
    explicit = re.findall(r"#(\w+)", text)
    if explicit:
        seen: dict[str, None] = {}
        for tag in explicit:
            tag_lower = tag.lower()
            if len(tag_lower) >= _MIN_HASHTAG_LEN:
                seen[tag_lower] = None
        return list(seen)

    # 2. Implicit fallback — proper nouns + acronyms only
    # Acronyms: ALL-CAPS, ≥ 2 characters (NFL, FBI, CIA, POTUS, …)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)

    # Proper nouns: Title-case word, ≥ 4 chars, not a common stopword
    title_case = re.findall(r"\b[A-Z][a-z]{3,}\b", text)

    implicit: dict[str, None] = {}
    for word in acronyms:
        implicit[word.lower()] = None

    for word in title_case:
        w = word.lower()
        if w not in _STOPWORDS:
            implicit[w] = None

    return list(implicit)


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Build partial M3DUSA heterogeneous graphs (News + Hashtag) from text.

    The graph schema matches the M3DUSA ``HeteroData`` shape so that the same
    ``HGTConv`` / ``HANConv`` graph encoder can be used for both the
    API-free training run and a full-schema run (once Twitter data is available).

    Args:
        model_name: ``sentence-transformers`` model identifier for node
                    feature computation.
    """

    def __init__(self, model_name: str = _DEFAULT_EMBEDDER) -> None:
        self.model_name = model_name
        self._model = None   # lazy-loaded on first embed call
        log.info(f"GraphBuilder (HeteroData, News+Hashtag) configured — "
                 f"embedder={model_name}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _embedder(self):
        """Lazy-load the sentence-transformers model on first access."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            log.info(f"Loading sentence embedder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            log.info("Sentence embedder loaded.")
        return self._model

    def _embed(self, texts: list[str]) -> torch.Tensor:
        """Return unit-normalised sentence embeddings, shape ``(N, D)``."""
        return self._embedder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).cpu().float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, title: str) -> HeteroData:
        """Build a ``HeteroData`` graph for a single claim *title*.

        Graph schema::

            node types : "news"  (1 node)
                         "hashtag"  (H nodes, H ≥ 0)
            edge types : ("news", "has_hashtag", "hashtag")
                         ("hashtag", "rev_has_hashtag", "news")  ← reverse

        When *title* contains no extractable hashtags, the graph has a single
        ``news`` node and **no hashtag nodes or edges** — the ``HeteroData``
        object still has the correct keys with empty tensors so downstream
        code doesn't need special-casing.

        Args:
            title: Cleaned claim title string.

        Returns:
            ``torch_geometric.data.HeteroData`` with node feature matrices
            and edge indices in COO format.
        """
        graph = HeteroData()

        # --- News node (always exactly 1) ---
        news_emb = self._embed([title])          # (1, D)
        graph["news"].x = news_emb               # (1, D)

        # --- Hashtag extraction ---
        tags = extract_hashtags(title)

        if not tags:
            # No hashtags — empty hashtag nodes, empty edges
            D = news_emb.size(1)
            graph["hashtag"].x = torch.zeros((0, D), dtype=torch.float)
            graph["news", "has_hashtag",     "hashtag"].edge_index = \
                torch.zeros((2, 0), dtype=torch.long)
            graph["hashtag", "rev_has_hashtag", "news"].edge_index = \
                torch.zeros((2, 0), dtype=torch.long)
            log.debug(f"No hashtags found in: '{title[:60]}…'")
            return graph

        # --- Hashtag node features ---
        hash_emb = self._embed(tags)             # (H, D)
        graph["hashtag"].x = hash_emb

        H = len(tags)

        # --- Edges: news → hashtag (one edge per hashtag) ---
        # news node index 0 connects to all hashtag nodes 0..H-1
        src_n2h = torch.zeros(H, dtype=torch.long)          # [0, 0, ..., 0]
        dst_n2h = torch.arange(H, dtype=torch.long)         # [0, 1, ..., H-1]
        graph["news", "has_hashtag", "hashtag"].edge_index = \
            torch.stack([src_n2h, dst_n2h], dim=0)           # (2, H)

        # --- Reverse edges: hashtag → news (for undirected message passing) ---
        graph["hashtag", "rev_has_hashtag", "news"].edge_index = \
            torch.stack([dst_n2h, src_n2h], dim=0)           # (2, H)

        log.debug(f"Built HeteroData: 1 news node, {H} hashtag nodes, "
                  f"{H} has_hashtag edges — title='{title[:50]}…'")
        return graph

    def metadata(self) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Return the static graph metadata required by ``HGTConv`` / ``HANConv``.

        Returns:
            ``(node_types, edge_types)`` where *edge_types* are
            ``(src_type, relation, dst_type)`` triples.

        Example::

            node_types, edge_types = builder.metadata()
            # node_types: ["news", "hashtag"]
            # edge_types: [("news","has_hashtag","hashtag"),
            #              ("hashtag","rev_has_hashtag","news")]
        """
        node_types = ["news", "hashtag"]
        edge_types = [
            ("news",    "has_hashtag",      "hashtag"),
            ("hashtag", "rev_has_hashtag",  "news"),
        ]
        return node_types, edge_types
