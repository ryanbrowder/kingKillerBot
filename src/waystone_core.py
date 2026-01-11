from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama_client import ollama_chat


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
_RERANKER: Optional[CrossEncoder] = None


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


def get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _RERANKER


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
    """
    index = get_index()
    meta = get_meta()
    embedder = get_embedder()

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Get initial candidates
    scores, ids = index.search(q_emb, search_k)
    
    candidates = []
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
            candidates.append(out)
        
        if debug:
            preview = (row.get("text", "")[:220]).replace("\n", " ")
            dbg_rows.append({
                "rank": rank,
                "idx": idx,
                "score": float(score),
                "chapter": chapter,
                "passes_chapter_filter": passes,
                "chunk_id": row.get("chunk_id"),
                "preview": preview,
            })
    
    # Rerank ALL candidates that passed the filter
    if len(candidates) > 1:
        reranker = get_reranker()
        texts = [c.get("text", "") for c in candidates]
        pairs = [[query, text] for text in texts]
        rerank_scores = reranker.predict(pairs)
        
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)
        
        candidates.sort(key=lambda x: x.get("rerank_score", -999), reverse=True)
    
    # Take top k after reranking
    results = candidates[:k]
    
    if debug:
        # Update debug rows
        result_ids = {r["idx"] for r in results}
        for row in dbg_rows:
            if row["idx"] in result_ids:
                matching = next((r for r in results if r["idx"] == row["idx"]), None)
                if matching:
                    row["rerank_score"] = matching.get("rerank_score")
                    row["in_final_results"] = True
            else:
                row["in_final_results"] = False
        
        return results, {
            "query": query,
            "max_chapter": int(max_chapter),
            "k": int(k),
            "final_search_k": int(search_k),
            "returned": len(results),
            "total_candidates": len(candidates),
            "rows": dbg_rows,
        }
    
    return results


def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 28000) -> str:
    """
    Formats retrieved chunks into a compact context block.
    Caps total chars to keep prompts tight.
    """
    parts = []
    total = 0
    for c in chunks:
        header = f"[{c.get('chunk_id')} | Chapter {c.get('chapter')}]\n"
        text = (c.get("text") or "").strip()
        block = header + text
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def answer(question: str, current_chapter: int, model: str) -> Dict[str, Any]:
    """
    Full RAG answer: retrieve chunks (<= spoiler boundary) then ask Ollama.
    Returns {answer, sources, debug}.
    """
    # Retrieve
    if STATE.get("debug"):
        chunks, dbg = retrieve_chunks(
            query=question,
            max_chapter=int(current_chapter),
            k=16,
            search_k=500,
            debug=True,
        )
    else:
        chunks = retrieve_chunks(
            query=question,
            max_chapter=int(current_chapter),
            k=16,
            search_k=500,
            debug=False,
        )
        dbg = None

    context = _format_context(chunks)

    system = (
        "You are The Waystone Companion, a knowledgeable guide to The Name of the Wind.\n\n"
        
        "CORE RULES:\n"
        "1. Answer ONLY using the provided context chunks. If the answer isn't in the context, say so explicitly.\n"
        "2. NEVER reference events, characters, or plot points beyond the spoiler boundary (chapter limit).\n"
        "3. When the context has conflicting information or gaps, acknowledge the ambiguity.\n"
        "4. Use 1-2 brief quotes when they directly support your answer, but prioritize your own synthesis.\n\n"
        
        "RESPONSE STYLE:\n"
        "- Be concise but complete (2-4 paragraphs max)\n"
        "- Speak naturally, not like a wiki\n"
        "- If asked about something not yet revealed: 'That hasn't been explained by chapter X' or 'The text doesn't say yet'\n"
        "- If context is thin: 'Based on the limited context, [tentative answer], but this might be clarified later'\n\n"
        
        "EXAMPLE GOOD ANSWER:\n"
        "Q: 'Who is Denna?'\n"
        "A: 'Denna is a young woman Kvothe meets and becomes infatuated with. She's described as beautiful and elusive, "
        "often disappearing without explanation. The text notes she seems to have multiple patrons and goes by different names. "
        "By chapter 9, their relationship is just beginning and much about her remains mysterious.'\n\n"
        
        "EXAMPLE BAD ANSWER:\n"
        "Q: 'Who is Denna?'\n"
        "A: 'Denna is Kvothe's love interest and the most important woman in his life. She has a troubled past and...' "
        "[BAD: Makes claims not supported by early chapters, sounds like a wiki summary]\n"
    )

    user = (
        f"SPOILER BOUNDARY: Up to and including Chapter {int(current_chapter)}\n\n"
        f"CONTEXT FROM THE BOOK:\n{context}\n\n"
        f"---\n\n"
        f"READER'S QUESTION: {question}\n\n"
        f"Your answer (ground everything in the context above, respect the chapter {int(current_chapter)} boundary):"
    )

    # Call Ollama
    llm_text = ollama_chat(model=model, system=system, user=user)

    sources = [
        {"chunk_id": c.get("chunk_id"), "chapter": c.get("chapter"), "score": c.get("score")}
        for c in chunks
    ]

    return {"answer": llm_text, "sources": sources, "debug": dbg}