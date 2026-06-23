import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


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


def dense_search(query: str, model, index, chunks: list, top_k: int = 10):
    query_embedding = model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(query_embedding, top_k)

    results = []

    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        idx = int(idx)
        c = chunks[idx]
        metadata = c.get("metadata", {})

        results.append({
            "rank": rank,
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
        help="Thư mục lưu dense index và report",
    )

    parser.add_argument(
        "--chunk_file_name",
        type=str,
        default="legal_chunks_final.jsonl",
        help="Tên file chunk chính",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Tên embedding model trên HuggingFace",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size encode embedding. T4 nên dùng 8 hoặc 16.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Độ dài tối đa token cho embedding model",
    )

    parser.add_argument(
        "--test_query",
        type=str,
        default="Hồ sơ đăng ký doanh nghiệp bằng tiếng nước ngoài có cần dịch công chứng không?",
        help="Câu hỏi test dense retrieval",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Số kết quả test dense retrieval",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Input dir    :", input_dir)
    print("[INFO] Artifact dir :", artifact_dir)
    print("[INFO] Model name   :", args.model_name)

    chunk_path = find_chunk_file(input_dir, args.chunk_file_name)
    print("[INFO] Chunk path:", chunk_path)

    chunks = load_jsonl(chunk_path)
    texts = [get_search_text(c) for c in chunks]

    empty_texts = sum(1 for t in texts if not t.strip())

    print("[INFO] Total chunks:", len(chunks))
    print("[INFO] Empty texts :", empty_texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] CUDA device count:", torch.cuda.device_count())
    print("[INFO] Using device:", device)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(args.model_name, device=device)

    # Giảm max length để tránh tràn VRAM trên T4.
    model.max_seq_length = args.max_seq_length

    print("[INFO] Encoding texts...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    print("[INFO] Embeddings shape:", embeddings.shape)

    print("[INFO] Building FAISS index...")
    dim = embeddings.shape[1]

    # Vì embedding đã normalize, dùng Inner Product tương đương cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    dense_index_path = artifact_dir / "dense_faiss.index"
    embeddings_path = artifact_dir / "dense_embeddings.npy"
    chunks_path = artifact_dir / "chunks.pkl"

    faiss.write_index(index, str(dense_index_path))
    np.save(embeddings_path, embeddings)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    build_report = {
        "chunk_path": str(chunk_path),
        "total_chunks": len(chunks),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "embedding_dim": int(dim),
        "dense_index_path": str(dense_index_path),
        "embeddings_path": str(embeddings_path),
        "chunks_path": str(chunks_path),
        "device": device,
        "cuda_device_count": torch.cuda.device_count(),
    }

    save_json(artifact_dir / "dense_build_report.json", build_report)

    print("\n========== DENSE BUILD REPORT ==========")
    print("Total chunks :", len(chunks))
    print("Embedding dim:", dim)
    print("Saved index  :", dense_index_path)
    print("Saved embeds :", embeddings_path)
    print("Saved chunks :", chunks_path)
    print("========================================\n")

    print("[TEST QUERY]", args.test_query)

    test_results = dense_search(
        query=args.test_query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=args.top_k,
    )

    save_json(artifact_dir / "dense_test_results.json", test_results)

    print("\n========== DENSE TEST RESULTS ==========")

    for r in test_results:
        print("-" * 100)
        print("Rank      :", r["rank"])
        print("Score     :", r["score"])
        print("Citation  :", r["citation"])
        print("Title     :", r["retrieval_title"])
        print("Ref keys  :", r["legal_reference_keys"])
        print("Preview   :", r["content_preview"][:300])

    print("========================================\n")
    print("[DONE] Dense embedding index built successfully.")


if __name__ == "__main__":
    main()