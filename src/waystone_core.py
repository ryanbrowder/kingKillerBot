from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer


# ------------------
# Global state
# ------------------
STATE = {
    "book": "NOTW",
    "chapter": 9,
    "model": "mistral",
    "debug": False,
    "show_sources": True,
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

FAISS_PATH = INDEX_DIR / "NOTW.faiss"
META_PATH = INDEX_DIR / "NOTW_meta.jsonl"

# Must match whatever you used when building the FAISS index embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Streamlit reruns your script a lot; cache heavy objects in module globals
_INDEX: Optional[faiss.Index] = None
_META: Optional[List[Dict[str, Any]]] = None
_EMBED: Optional[SentenceTransformer] = None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_index() -> faiss.Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(str(FAISS_PATH))
    return _INDEX


def get_meta() -> List[Dict[str, Any]]:
    global _META
    if _META is None:
        _META = _read_jsonl(META_PATH)
    return _META


def get_embedder() -> SentenceTransformer:
    global _EMBED
    if _EMBED is None:
        _EMBED = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED


def retrieve_chunks(
    query: str,
    max_chapter: int,
    k: int = 12,
    search_k: int = 80,
    max_search_k: int = 500,
    debug: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve up to k chunks relevant to the query, filtered by chapter <= max_chapter.

    Returns: List[dict] where each dict includes at least:
      - book, chapter, chunk_id, text
      - plus: score, idx (added by this function)

    If debug=True, returns (results, debug_payload).
    """
    index = get_index()
    meta = get_meta()
    embedder = get_embedder()

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    results: List[Dict[str, Any]] = []
    dbg_rows: List[Dict[str, Any]] = []

    cur_search_k = max(search_k, k)

    while True:
        scores, ids = index.search(q_emb, cur_search_k)

        results = []
        dbg_rows = []

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            idx = int(idx)
            row = meta[idx]

            chapter = int(row.get("chapter", 10**9))
            passes = chapter <= int(max_chapter)

            if passes:
                out = dict(row)
                out["score"] = float(score)
                out["idx"] = idx
                results.append(out)

            if debug:
                preview = (row.get("text", "")[:220]).replace("\n", " ")
                dbg_rows.append(
                    {
                        "rank": rank,
                        "idx": idx,
                        "score": float(score),
                        "chapter": chapter,
                        "passes_chapter_filter": passes,
                        "chunk_id": row.get("chunk_id"),
                        "preview": preview,
                    }
                )

            if len(results) >= k:
                break

        if len(results) >= k or cur_search_k >= max_search_k:
            break

        cur_search_k = min(cur_search_k * 2, max_search_k)

    if debug:
        return results, {
            "query": query,
            "max_chapter": int(max_chapter),
            "k": int(k),
            "final_search_k": int(cur_search_k),
            "returned": len(results),
            "rows": dbg_rows,
        }

    return results