import argparse
import json
import pickle
import re
from pathlib import Path
from collections import Counter

from rank_bm25 import BM25Okapi


def find_chunk_file(input_dir: Path, chunk_file_name: str) -> Path:
    candidates = list(input_dir.rglob(chunk_file_name))

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy {chunk_file_name} trong {input_dir}"
        )

    if len(candidates) > 1:
        print("[WARNING] Tìm thấy nhiều file chunk:")
        for c in candidates:
            print(" -", c)
        print("[INFO] Sẽ dùng file đầu tiên:", candidates[0])

    return candidates[0]


def load_jsonl(path: Path) -> list:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Lỗi JSON ở dòng {line_no}: {e}")

    return rows


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list:
    text = normalize_text(text)

    # Giữ số hiệu văn bản kiểu 123/2020/nđ-cp, điều/khoản/số, và từ tiếng Việt
    tokens = re.findall(
        r"\d+/\d{4}/[a-zà-ỹđ\-]+|[a-zà-ỹđ]+|\d+",
        text,
        flags=re.IGNORECASE,
    )

    return tokens


def get_search_text(chunk: dict) -> str:
    search_text = chunk.get("search_text")
    if search_text:
        return search_text

    chunk_text = chunk.get("chunk_text")
    if chunk_text:
        return chunk_text

    content = chunk.get("content")
    if content:
        return content

    return ""


def build_bm25(chunks: list):
    texts = [get_search_text(c) for c in chunks]
    tokenized_corpus = [tokenize(t) for t in texts]

    empty_token_rows = sum(1 for row in tokenized_corpus if not row)

    print("[INFO] Total chunks:", len(chunks))
    print("[INFO] Empty token rows:", empty_token_rows)

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


def bm25_search(query: str, chunks: list, bm25, top_k: int = 10):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for idx, score in ranked:
        c = chunks[idx]
        metadata = c.get("metadata", {})

        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "chunk_id": c.get("chunk_id"),
            "parent_article_id": c.get("parent_article_id"),
            "chunk_level": c.get("chunk_level"),
            "citation": c.get("citation"),
            "retrieval_title": c.get("retrieval_title"),
            "document_number": metadata.get("document_number"),
            "document_title": metadata.get("document_title"),
            "document_type": metadata.get("document_type"),
            "legal_reference_keys": metadata.get("legal_reference_keys", []),
            "content_preview": c.get("content", "")[:500],
        })

    return results


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/kaggle/input/legal-v3/chunk_v3",
        help="Thư mục chứa legal_chunks_final.jsonl",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
        help="Thư mục lưu BM25 index và report",
    )

    parser.add_argument(
        "--chunk_file_name",
        type=str,
        default="legal_chunks_final.jsonl",
        help="Tên file chunk chính",
    )

    parser.add_argument(
        "--test_query",
        type=str,
        default="Hồ sơ đăng ký doanh nghiệp bằng tiếng nước ngoài có cần dịch công chứng không?",
        help="Câu hỏi test BM25",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Số kết quả test BM25",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Input dir    :", input_dir)
    print("[INFO] Artifact dir :", artifact_dir)

    chunk_path = find_chunk_file(input_dir, args.chunk_file_name)
    print("[INFO] Chunk path:", chunk_path)

    chunks = load_jsonl(chunk_path)

    bm25, tokenized_corpus = build_bm25(chunks)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    tokenized_path = artifact_dir / "tokenized_corpus.pkl"

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    with open(tokenized_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    chunk_level_counter = Counter(c.get("chunk_level", "UNKNOWN") for c in chunks)
    document_type_counter = Counter(
        c.get("metadata", {}).get("document_type", "UNKNOWN")
        for c in chunks
    )

    build_report = {
        "chunk_path": str(chunk_path),
        "total_chunks": len(chunks),
        "bm25_path": str(bm25_path),
        "chunks_path": str(chunks_path),
        "tokenized_corpus_path": str(tokenized_path),
        "chunk_level_distribution": dict(chunk_level_counter),
        "document_type_distribution": dict(document_type_counter),
    }

    save_json(artifact_dir / "bm25_build_report.json", build_report)

    print("\n========== BM25 BUILD REPORT ==========")
    print("Total chunks:", len(chunks))
    print("Saved BM25  :", bm25_path)
    print("Saved chunks:", chunks_path)
    print("=======================================\n")

    print("[TEST QUERY]", args.test_query)

    test_results = bm25_search(
        query=args.test_query,
        chunks=chunks,
        bm25=bm25,
        top_k=args.top_k,
    )

    save_json(artifact_dir / "bm25_test_results.json", test_results)

    print("\n========== BM25 TEST RESULTS ==========")

    for r in test_results:
        print("-" * 100)
        print("Rank      :", r["rank"])
        print("Score     :", r["score"])
        print("Citation  :", r["citation"])
        print("Title     :", r["retrieval_title"])
        print("Ref keys  :", r["legal_reference_keys"])
        print("Preview   :", r["content_preview"][:300])

    print("=======================================\n")
    print("[DONE] BM25 baseline built successfully.")


if __name__ == "__main__":
    main()