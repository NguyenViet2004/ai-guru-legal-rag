import argparse
import importlib.util
import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str, max_chars: int = 5000) -> str:
    if not text:
        return ""

    text = " ".join(str(text).split())
    return text[:max_chars]


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)

    if abs(v_max - v_min) < 1e-12:
        return [1.0 for _ in values]

    return [(v - v_min) / (v_max - v_min) for v in values]


def build_chunk_lookup(chunks: list[dict]):
    by_chunk_id = {}
    by_parent_article_id = {}
    by_title = {}
    by_ref = {}

    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}

        chunk_id = chunk.get("chunk_id")
        parent_id = chunk.get("parent_article_id")
        title = chunk.get("retrieval_title")
        refs = metadata.get("legal_reference_keys", []) or []

        if chunk_id:
            by_chunk_id[chunk_id] = chunk

        if parent_id and parent_id not in by_parent_article_id:
            by_parent_article_id[parent_id] = chunk

        if title and title not in by_title:
            by_title[title] = chunk

        for ref in refs:
            if ref not in by_ref:
                by_ref[ref] = chunk

    return {
        "by_chunk_id": by_chunk_id,
        "by_parent_article_id": by_parent_article_id,
        "by_title": by_title,
        "by_ref": by_ref,
    }


def find_original_chunk(result: dict, lookup: dict):
    chunk_id = result.get("chunk_id")
    parent_id = result.get("parent_article_id")
    title = result.get("retrieval_title")
    refs = result.get("legal_reference_keys", []) or []

    if chunk_id and chunk_id in lookup["by_chunk_id"]:
        return lookup["by_chunk_id"][chunk_id]

    if parent_id and parent_id in lookup["by_parent_article_id"]:
        return lookup["by_parent_article_id"][parent_id]

    if title and title in lookup["by_title"]:
        return lookup["by_title"][title]

    for ref in refs:
        if ref in lookup["by_ref"]:
            return lookup["by_ref"][ref]

    return None


def get_full_content(result: dict, lookup: dict, max_chars: int):
    original_chunk = find_original_chunk(result, lookup)

    if original_chunk:
        content = (
            original_chunk.get("content")
            or original_chunk.get("chunk_text")
            or original_chunk.get("search_text")
            or ""
        )
        return clean_text(content, max_chars=max_chars)

    return clean_text(
        result.get("content")
        or result.get("content_preview")
        or result.get("chunk_text")
        or "",
        max_chars=max_chars,
    )


def build_passage_for_reranker(result: dict, lookup: dict, max_chars: int = 3500) -> str:
    title = result.get("retrieval_title") or ""
    citation = result.get("citation") or ""
    content = get_full_content(result, lookup, max_chars=max_chars)

    parts = []

    if title:
        parts.append(f"Tiêu đề: {title}")

    if citation:
        parts.append(f"Căn cứ: {citation}")

    if content:
        parts.append(f"Nội dung: {content}")

    return "\n".join(parts)


def rerank_with_base_model(
    query: str,
    results: list[dict],
    lookup: dict,
    reranker_model_name: str,
    device: str,
    max_chars: int,
):
    print("[INFO] Loading reranker:", reranker_model_name)

    reranker = CrossEncoder(
        reranker_model_name,
        max_length=512,
        device=device,
    )

    pairs = []

    for result in results:
        passage = build_passage_for_reranker(
            result=result,
            lookup=lookup,
            max_chars=max_chars,
        )
        pairs.append([query, passage])

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    for result, score in zip(results, scores):
        result["reranker_score"] = float(score)

    sorted_by_reranker = sorted(
        results,
        key=lambda x: x.get("reranker_score", 0.0),
        reverse=True,
    )

    for rank, result in enumerate(sorted_by_reranker, start=1):
        result["reranker_rank"] = rank

    return results


def blend_ranking(
    results: list[dict],
    hybrid_weight: float = 0.75,
    reranker_weight: float = 0.25,
):
    """
    Không để reranker override hoàn toàn Hybrid.
    Dùng reciprocal-rank blend:
    - Hybrid vẫn là chính
    - Reranker chỉ hỗ trợ điều chỉnh nhẹ
    """

    for result in results:
        hybrid_rank = result.get("rank", 9999)
        reranker_rank = result.get("reranker_rank", hybrid_rank)

        hybrid_rank_score = 1.0 / max(hybrid_rank, 1)
        reranker_rank_score = 1.0 / max(reranker_rank, 1)

        result["hybrid_rank_score"] = hybrid_rank_score
        result["reranker_rank_score"] = reranker_rank_score

        result["final_context_score"] = (
            hybrid_weight * hybrid_rank_score
            + reranker_weight * reranker_rank_score
        )

    blended = sorted(
        results,
        key=lambda x: x["final_context_score"],
        reverse=True,
    )

    for rank, result in enumerate(blended, start=1):
        result["final_context_rank"] = rank

    return blended


def deduplicate_contexts(results: list[dict], top_k: int):
    selected = []
    seen_parent = set()
    seen_ref = set()
    seen_title = set()

    for result in results:
        parent_id = result.get("parent_article_id")
        refs = tuple(result.get("legal_reference_keys", []) or [])
        title = result.get("retrieval_title")

        dedup_key = None

        if parent_id:
            dedup_key = ("parent", parent_id)
        elif refs:
            dedup_key = ("refs", refs)
        elif title:
            dedup_key = ("title", title)

        if dedup_key:
            if dedup_key[0] == "parent" and dedup_key[1] in seen_parent:
                continue
            if dedup_key[0] == "refs" and dedup_key[1] in seen_ref:
                continue
            if dedup_key[0] == "title" and dedup_key[1] in seen_title:
                continue

        if parent_id:
            seen_parent.add(parent_id)

        if refs:
            seen_ref.add(refs)

        if title:
            seen_title.add(title)

        selected.append(result)

        if len(selected) >= top_k:
            break

    return selected


def build_context_blocks(
    selected_results: list[dict],
    lookup: dict,
    max_context_chars: int,
):
    blocks = []

    for i, result in enumerate(selected_results, start=1):
        refs = result.get("legal_reference_keys", []) or []
        title = result.get("retrieval_title") or ""
        citation = result.get("citation") or ""
        content = get_full_content(
            result=result,
            lookup=lookup,
            max_chars=max_context_chars,
        )

        block = {
            "context_id": i,
            "legal_reference_keys": refs,
            "retrieval_title": title,
            "citation": citation,
            "content": content,
            "hybrid_rank": result.get("rank"),
            "reranker_rank": result.get("reranker_rank"),
            "final_context_rank": result.get("final_context_rank"),
            "reranker_score": result.get("reranker_score"),
            "final_context_score": result.get("final_context_score"),
        }

        blocks.append(block)

    return blocks


def build_prompt(question: str, context_blocks: list[dict]) -> str:
    context_texts = []

    for block in context_blocks:
        refs = ", ".join(block.get("legal_reference_keys", []) or [])
        title = block.get("retrieval_title") or ""
        citation = block.get("citation") or ""
        content = block.get("content") or ""

        context_texts.append(
            f"""[Căn cứ {block['context_id']}]
Mã căn cứ: {refs}
Tiêu đề: {title}
Trích dẫn: {citation}
Nội dung:
{content}
"""
        )

    joined_contexts = "\n".join(context_texts)

    prompt = f"""Bạn là trợ lý pháp lý cho doanh nghiệp tại Việt Nam.

Nhiệm vụ:
- Trả lời câu hỏi dựa trên các căn cứ pháp lý được cung cấp.
- Không được bịa nội dung ngoài căn cứ.
- Nếu căn cứ chưa đủ để kết luận, hãy nói rõ là chưa đủ căn cứ.
- Trả lời ngắn gọn, rõ ràng, dễ hiểu.
- Cuối câu trả lời phải nêu căn cứ pháp lý đã sử dụng.

[Câu hỏi]
{question}

[Căn cứ pháp lý]
{joined_contexts}

[Yêu cầu định dạng câu trả lời]
1. Trả lời trực tiếp câu hỏi.
2. Giải thích ngắn gọn theo căn cứ pháp lý.
3. Liệt kê căn cứ pháp lý ở cuối.
"""
    return prompt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="BAAI/bge-m3",
    )

    parser.add_argument(
        "--use_reranker",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
    )

    parser.add_argument(
        "--bm25_top_k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dense_top_k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--candidate_top_k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--context_top_k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--hybrid_weight",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--reranker_weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max_reranker_chars",
        type=int,
        default=3500,
    )

    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="/kaggle/working/artifacts/rag_context_prompt.json",
    )

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)
    output_file = Path(args.output_file)

    hybrid_module_path = root_dir / "scripts" / "04_hybrid_retrieval.py"
    hybrid = load_module_from_path("hybrid_retrieval", hybrid_module_path)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    print("[INFO] Question:", args.question)
    print("[INFO] Artifact dir:", artifact_dir)

    print("[INFO] Loading BM25:", bm25_path)
    bm25 = hybrid.load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = hybrid.load_pickle(chunks_path)

    print("[INFO] Building chunk lookup")
    lookup = build_chunk_lookup(chunks)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    print("[INFO] Loading embedding model:", args.embedding_model_name)
    dense_model = SentenceTransformer(
        args.embedding_model_name,
        device=device,
    )

    print("[INFO] Running hybrid search")

    hybrid_results = hybrid.hybrid_search(
        query=args.question,
        chunks=chunks,
        bm25=bm25,
        dense_model=dense_model,
        dense_index=dense_index,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.candidate_top_k,
    )

    print("[INFO] Hybrid candidates:", len(hybrid_results))

    results = [dict(r) for r in hybrid_results]

    if args.use_reranker:
        results = rerank_with_base_model(
            query=args.question,
            results=results,
            lookup=lookup,
            reranker_model_name=args.reranker_model_name,
            device=device,
            max_chars=args.max_reranker_chars,
        )

        results = blend_ranking(
            results,
            hybrid_weight=args.hybrid_weight,
            reranker_weight=args.reranker_weight,
        )
    else:
        for result in results:
            result["final_context_score"] = 1.0 / max(result.get("rank", 9999), 1)
            result["final_context_rank"] = result.get("rank")

        results = sorted(
            results,
            key=lambda x: x["final_context_score"],
            reverse=True,
        )

    selected = deduplicate_contexts(
        results=results,
        top_k=args.context_top_k,
    )

    context_blocks = build_context_blocks(
        selected_results=selected,
        lookup=lookup,
        max_context_chars=args.max_context_chars,
    )

    prompt = build_prompt(
        question=args.question,
        context_blocks=context_blocks,
    )

    output = {
        "question": args.question,
        "use_reranker": bool(args.use_reranker),
        "embedding_model_name": args.embedding_model_name,
        "reranker_model_name": args.reranker_model_name if args.use_reranker else None,
        "hybrid_weight": args.hybrid_weight,
        "reranker_weight": args.reranker_weight if args.use_reranker else None,
        "context_top_k": args.context_top_k,
        "contexts": context_blocks,
        "prompt": prompt,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_file, output)

    print("\n========== RAG CONTEXT RESULT ==========")
    print("Question:", args.question)
    print("Use reranker:", bool(args.use_reranker))
    print("Selected contexts:", len(context_blocks))
    print("Saved:", output_file)

    for block in context_blocks:
        print("-" * 100)
        print("Rank:", block["context_id"])
        print("Refs:", block["legal_reference_keys"])
        print("Title:", block["retrieval_title"])
        print("Hybrid rank:", block["hybrid_rank"])
        print("Reranker rank:", block["reranker_rank"])
        print("Final score:", block["final_context_score"])
        print("Content preview:", block["content"][:300])

    print("========================================\n")

    print("\n========== PROMPT PREVIEW ==========")
    print(prompt[:4000])
    print("====================================\n")


if __name__ == "__main__":
    main()