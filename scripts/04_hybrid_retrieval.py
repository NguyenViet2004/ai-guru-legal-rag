import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list:
    text = normalize_text(text)

    tokens = re.findall(
        r"\d+/\d{4}/[a-zà-ỹđ\-]+|[a-zà-ỹđ]+|\d+",
        text,
        flags=re.IGNORECASE,
    )

    return tokens


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_chunk_text(chunk: dict) -> str:
    return (
        chunk.get("search_text")
        or chunk.get("chunk_text")
        or chunk.get("content")
        or ""
    )


def detect_query_signals(query: str) -> dict:
    query_lower = query.lower()
    query_upper = query.upper()

    document_numbers = re.findall(
        r"\d+/\d{4}/[A-ZÀ-ỸĐ\-]+",
        query_upper,
        flags=re.IGNORECASE,
    )

    articles = re.findall(
        r"(?:điều|dieu)\s+(\d+[a-zA-Z]?)",
        query_lower,
        flags=re.IGNORECASE,
    )

    clauses = re.findall(
        r"(?:khoản|khoan)\s+(\d+[a-zA-Z]?)",
        query_lower,
        flags=re.IGNORECASE,
    )

    return {
        "document_numbers": [d.upper() for d in document_numbers],
        "articles": [a.lower() for a in articles],
        "clauses": [c.lower() for c in clauses],
    }


def get_metadata_value(metadata: dict, keys: List[str], default=None):
    for key in keys:
        if key in metadata and metadata[key] not in [None, "", []]:
            return metadata[key]
    return default


def rule_boost(query: str, chunk: dict, query_signals: dict) -> float:
    boost = 0.0

    metadata = chunk.get("metadata", {})
    query_lower = query.lower()
    query_upper = query.upper()

    document_number = str(
        get_metadata_value(
            metadata,
            ["document_number", "doc_number", "so_hieu_van_ban"],
            "",
        )
    ).upper()

    article_number = str(
        get_metadata_value(
            metadata,
            ["article_number", "article_id", "dieu"],
            "",
        )
    ).lower()

    legal_reference_keys = metadata.get("legal_reference_keys", []) or []
    citation_aliases = metadata.get("citation_aliases", []) or []
    plain_language_aliases = metadata.get("plain_language_aliases", []) or []

    citation = chunk.get("citation", "") or ""
    retrieval_title = chunk.get("retrieval_title", "") or ""

    combined_refs = " ".join(
        [
            document_number,
            citation,
            retrieval_title,
            " ".join(legal_reference_keys),
            " ".join(citation_aliases),
            " ".join(plain_language_aliases),
        ]
    ).upper()

    # Boost nếu query nêu rõ số hiệu văn bản
    for doc_num in query_signals["document_numbers"]:
        if doc_num and doc_num in combined_refs:
            boost += 2.5

    # Boost nếu query nêu rõ Điều
    for article in query_signals["articles"]:
        if article and article == article_number:
            boost += 1.5

        if f"ĐIỀU {article.upper()}" in combined_refs:
            boost += 1.0

    # Boost nhẹ nếu title/citation chứa nhiều từ khóa domain rõ
    important_terms = [
        "đăng ký doanh nghiệp",
        "hồ sơ đăng ký doanh nghiệp",
        "hóa đơn điện tử",
        "bảo hiểm xã hội",
        "hợp đồng lao động",
        "dữ liệu cá nhân",
        "an ninh mạng",
        "đấu thầu",
        "đất đai",
        "môi trường",
        "an toàn thực phẩm",
    ]

    title_lower = f"{citation} {retrieval_title}".lower()

    for term in important_terms:
        if term in query_lower and term in title_lower:
            boost += 0.8

    return boost


def bm25_retrieve(query: str, bm25, top_k: int) -> List[Tuple[int, float]]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [(int(idx), float(score)) for idx, score in ranked]


def dense_retrieve(query: str, model, index, top_k: int) -> List[Tuple[int, float]]:
    query_embedding = model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(query_embedding, top_k)

    return [
        (int(idx), float(score))
        for idx, score in zip(ids[0], scores[0])
    ]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    chunks: list,
    bm25,
    dense_model,
    dense_index,
    bm25_top_k: int = 100,
    dense_top_k: int = 100,
    final_top_k: int = 15,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    group_by_parent: bool = True,
):
    bm25_results = bm25_retrieve(query, bm25, bm25_top_k)
    dense_results = dense_retrieve(query, dense_model, dense_index, dense_top_k)

    scores: Dict[int, dict] = {}

    for rank, (idx, raw_score) in enumerate(bm25_results, start=1):
        if idx not in scores:
            scores[idx] = {
                "idx": idx,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_raw": None,
                "dense_raw": None,
                "rrf_score": 0.0,
                "rule_boost": 0.0,
                "final_score": 0.0,
            }

        scores[idx]["bm25_rank"] = rank
        scores[idx]["bm25_raw"] = raw_score
        scores[idx]["rrf_score"] += bm25_weight * rrf_score(rank, rrf_k)

    for rank, (idx, raw_score) in enumerate(dense_results, start=1):
        if idx not in scores:
            scores[idx] = {
                "idx": idx,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_raw": None,
                "dense_raw": None,
                "rrf_score": 0.0,
                "rule_boost": 0.0,
                "final_score": 0.0,
            }

        scores[idx]["dense_rank"] = rank
        scores[idx]["dense_raw"] = raw_score
        scores[idx]["rrf_score"] += dense_weight * rrf_score(rank, rrf_k)

    query_signals = detect_query_signals(query)

    for idx, item in scores.items():
        chunk = chunks[idx]
        boost = rule_boost(query, chunk, query_signals)
        item["rule_boost"] = boost
        item["final_score"] = item["rrf_score"] + boost

    ranked_items = sorted(
        scores.values(),
        key=lambda x: x["final_score"],
        reverse=True,
    )

    if group_by_parent:
        grouped = {}

        for item in ranked_items:
            chunk = chunks[item["idx"]]
            parent_id = chunk.get("parent_article_id") or chunk.get("chunk_id")

            if parent_id not in grouped:
                grouped[parent_id] = item

        ranked_items = list(grouped.values())

    ranked_items = ranked_items[:final_top_k]

    output = []

    for rank, item in enumerate(ranked_items, start=1):
        chunk = chunks[item["idx"]]
        metadata = chunk.get("metadata", {})

        output.append({
            "rank": rank,
            "final_score": float(item["final_score"]),
            "rrf_score": float(item["rrf_score"]),
            "rule_boost": float(item["rule_boost"]),
            "bm25_rank": item["bm25_rank"],
            "dense_rank": item["dense_rank"],
            "bm25_raw": item["bm25_raw"],
            "dense_raw": item["dense_raw"],
            "chunk_id": chunk.get("chunk_id"),
            "parent_article_id": chunk.get("parent_article_id"),
            "chunk_level": chunk.get("chunk_level"),
            "citation": chunk.get("citation"),
            "retrieval_title": chunk.get("retrieval_title"),
            "document_number": metadata.get("document_number"),
            "document_title": metadata.get("document_title"),
            "document_type": metadata.get("document_type"),
            "legal_reference_keys": metadata.get("legal_reference_keys", []),
            "content_preview": chunk.get("content", "")[:700],
        })

    return output


def print_results(query: str, results: list):
    print("\n" + "=" * 120)
    print("[QUERY]", query)
    print("=" * 120)

    for r in results:
        print("-" * 120)
        print("Rank       :", r["rank"])
        print("Final score:", r["final_score"])
        print("RRF score  :", r["rrf_score"])
        print("Rule boost :", r["rule_boost"])
        print("BM25 rank  :", r["bm25_rank"], "| raw:", r["bm25_raw"])
        print("Dense rank :", r["dense_rank"], "| raw:", r["dense_raw"])
        print("Citation   :", r["citation"])
        print("Title      :", r["retrieval_title"])
        print("Ref keys   :", r["legal_reference_keys"])
        print("Preview    :", r["content_preview"][:350])

    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
        help="Thư mục chứa bm25.pkl, dense_faiss.index, chunks.pkl",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Tên dense embedding model",
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
        "--final_top_k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--test_query",
        type=str,
        action="append",
        default=None,
        help="Có thể truyền nhiều --test_query",
    )

    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    if not bm25_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {bm25_path}. Hãy chạy 02_build_bm25.py trước.")

    if not dense_index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {dense_index_path}. Hãy chạy 03_build_dense_index.py trước.")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {chunks_path}. Hãy chạy 02 hoặc 03 trước.")

    print("[INFO] Artifact dir:", artifact_dir)
    print("[INFO] Loading BM25:", bm25_path)
    bm25 = load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = load_pickle(chunks_path)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] Using device:", device)
    print("[INFO] Loading dense model:", args.model_name)

    dense_model = SentenceTransformer(args.model_name, device=device)

    queries = args.test_query or [
        "Hồ sơ đăng ký doanh nghiệp bằng tiếng nước ngoài có cần dịch công chứng không?",
        "Tài liệu tiếng Anh trong hồ sơ thành lập công ty có phải dịch sang tiếng Việt không?",
        "Người lao động bị sa thải trong trường hợp nào?",
        "Doanh nghiệp sử dụng hóa đơn điện tử khi nào?",
        "Quy định về xử lý dữ liệu cá nhân của doanh nghiệp là gì?",
    ]

    all_results = []

    for query in queries:
        results = hybrid_search(
            query=query,
            chunks=chunks,
            bm25=bm25,
            dense_model=dense_model,
            dense_index=dense_index,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            final_top_k=args.final_top_k,
        )

        print_results(query, results)

        all_results.append({
            "query": query,
            "results": results,
        })

    save_json(artifact_dir / "hybrid_test_results.json", all_results)

    print("[DONE] Saved hybrid test results to:", artifact_dir / "hybrid_test_results.json")


if __name__ == "__main__":
    main()