import argparse
import importlib.util
import json
import pickle
import random
from pathlib import Path
from statistics import mean

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path):
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


def save_jsonl(path: Path, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_chunk_text(chunk: dict) -> str:
    return (
        chunk.get("search_text")
        or chunk.get("chunk_text")
        or chunk.get("content")
        or ""
    )


def get_legal_refs_from_chunk(chunk: dict) -> set:
    metadata = chunk.get("metadata", {}) or {}
    refs = metadata.get("legal_reference_keys", []) or []
    return set(refs)


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def add_candidate(candidates: dict, idx: int, score: float, source: str):
    if idx not in candidates:
        candidates[idx] = {
            "idx": idx,
            "score": 0.0,
            "sources": [],
        }

    candidates[idx]["score"] += score
    candidates[idx]["sources"].append(source)


def is_valid_negative(
    candidate_chunk: dict,
    pair: dict,
    min_negative_len: int,
) -> bool:
    candidate_chunk_id = candidate_chunk.get("chunk_id")
    candidate_parent_id = candidate_chunk.get("parent_article_id")

    positive_chunk_id = pair.get("positive_chunk_id")
    positive_parent_id = pair.get("parent_article_id")

    if not candidate_chunk_id:
        return False

    # Không lấy chính positive
    if candidate_chunk_id == positive_chunk_id:
        return False

    # Không lấy chunk cùng parent article
    if positive_parent_id and candidate_parent_id and positive_parent_id == candidate_parent_id:
        return False

    negative_text = get_chunk_text(candidate_chunk)

    if len(negative_text.strip()) < min_negative_len:
        return False

    # Không lấy chunk có cùng legal_reference_keys
    positive_refs = set(pair.get("legal_reference_keys", []) or [])
    candidate_refs = get_legal_refs_from_chunk(candidate_chunk)

    if positive_refs and candidate_refs and positive_refs.intersection(candidate_refs):
        return False

    return True


def make_negative_item(candidate_chunk: dict, score_item: dict):
    metadata = candidate_chunk.get("metadata", {}) or {}

    return {
        "negative_chunk_id": candidate_chunk.get("chunk_id"),
        "negative_parent_article_id": candidate_chunk.get("parent_article_id"),
        "negative_text": get_chunk_text(candidate_chunk),
        "negative_content": candidate_chunk.get("content", ""),
        "negative_citation": candidate_chunk.get("citation"),
        "negative_retrieval_title": candidate_chunk.get("retrieval_title"),
        "negative_legal_reference_keys": metadata.get("legal_reference_keys", []) or [],
        "negative_document_number": metadata.get("document_number"),
        "negative_document_title": metadata.get("document_title"),
        "negative_document_type": metadata.get("document_type"),
        "mine_score": float(score_item["score"]),
        "mine_sources": score_item["sources"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
    )

    parser.add_argument(
        "--pairs_file",
        type=str,
        default="synthetic_train_pairs_v2.jsonl",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="hard_negative_train_v1.jsonl",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=20000,
        help="Số pair dùng để mine. 0 nghĩa là dùng toàn bộ.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--bm25_top_k",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--dense_top_k",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--num_negatives",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min_negative_len",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)

    pairs_path = artifact_dir / args.pairs_file
    output_path = artifact_dir / args.output_file
    stats_path = artifact_dir / "hard_negative_mining_stats_v1.json"
    sample_path = artifact_dir / "hard_negative_samples_v1.json"

    hybrid_module_path = root_dir / "scripts" / "04_hybrid_retrieval.py"
    hybrid = load_module_from_path("hybrid_retrieval", hybrid_module_path)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    if not pairs_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {pairs_path}")

    if not bm25_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {bm25_path}. Hãy chạy 02_build_bm25.py trước.")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {chunks_path}. Hãy chạy 02 hoặc 03 trước.")

    if not dense_index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {dense_index_path}. Hãy chạy 03_build_dense_index.py trước.")

    print("[INFO] Artifact dir:", artifact_dir)
    print("[INFO] Loading pairs:", pairs_path)

    pairs = load_jsonl(pairs_path)

    random.seed(args.seed)
    random.shuffle(pairs)

    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    print("[INFO] Pairs to mine:", len(pairs))

    print("[INFO] Loading BM25:", bm25_path)
    bm25 = load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = load_pickle(chunks_path)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading dense model:", args.model_name)
    print("[INFO] Device:", device)

    dense_model = SentenceTransformer(args.model_name, device=device)

    mined_rows = []
    skipped_no_negative = 0
    negative_counts = []

    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Mining hard negatives"):
        batch = pairs[start:start + args.batch_size]
        queries = [p["query"] for p in batch]

        # Dense search theo batch
        query_embeddings = dense_model.encode(
            queries,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        dense_scores_batch, dense_ids_batch = dense_index.search(
            query_embeddings,
            args.dense_top_k,
        )

        for i, pair in enumerate(batch):
            query = pair["query"]
            expanded_query = hybrid.expand_query(query)

            candidates = {}

            # Dense candidates
            dense_ids = dense_ids_batch[i]
            dense_scores = dense_scores_batch[i]

            for rank, (idx, raw_score) in enumerate(zip(dense_ids, dense_scores), start=1):
                idx = int(idx)

                if idx < 0:
                    continue

                add_candidate(
                    candidates,
                    idx,
                    rrf_score(rank),
                    f"dense_rank_{rank}_score_{float(raw_score):.4f}",
                )

            # BM25 candidates
            bm25_results = hybrid.bm25_retrieve(
                expanded_query,
                bm25,
                args.bm25_top_k,
            )

            for rank, (idx, raw_score) in enumerate(bm25_results, start=1):
                add_candidate(
                    candidates,
                    int(idx),
                    rrf_score(rank),
                    f"bm25_rank_{rank}_score_{float(raw_score):.4f}",
                )

            ranked_candidates = sorted(
                candidates.values(),
                key=lambda x: x["score"],
                reverse=True,
            )

            negatives = []

            for score_item in ranked_candidates:
                candidate_chunk = chunks[score_item["idx"]]

                if not is_valid_negative(
                    candidate_chunk=candidate_chunk,
                    pair=pair,
                    min_negative_len=args.min_negative_len,
                ):
                    continue

                negatives.append(make_negative_item(candidate_chunk, score_item))

                if len(negatives) >= args.num_negatives:
                    break

            if not negatives:
                skipped_no_negative += 1
                continue

            out_item = {
                "query": pair["query"],
                "positive_chunk_id": pair.get("positive_chunk_id"),
                "positive_parent_article_id": pair.get("parent_article_id"),
                "positive_text": pair.get("positive_text"),
                "positive_content": pair.get("positive_content"),
                "positive_citation": pair.get("citation"),
                "positive_retrieval_title": pair.get("retrieval_title"),
                "positive_legal_reference_keys": pair.get("legal_reference_keys", []) or [],
                "positive_document_number": pair.get("document_number"),
                "positive_document_title": pair.get("document_title"),
                "positive_document_type": pair.get("document_type"),
                "negatives": negatives,
                "source": "hard_negative_mining_v1",
            }

            mined_rows.append(out_item)
            negative_counts.append(len(negatives))

    save_jsonl(output_path, mined_rows)

    stats = {
        "pairs_file": str(pairs_path),
        "output_file": str(output_path),
        "total_pairs_input_used": len(pairs),
        "rows_with_negatives": len(mined_rows),
        "skipped_no_negative": skipped_no_negative,
        "num_negatives_target": args.num_negatives,
        "total_negative_items": sum(negative_counts),
        "avg_negatives_per_query": mean(negative_counts) if negative_counts else 0,
        "min_negatives_per_query": min(negative_counts) if negative_counts else 0,
        "max_negatives_per_query": max(negative_counts) if negative_counts else 0,
        "bm25_top_k": args.bm25_top_k,
        "dense_top_k": args.dense_top_k,
        "model_name": args.model_name,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
    }

    save_json(stats_path, stats)
    save_json(sample_path, mined_rows[:20])

    print("\n========== HARD NEGATIVE MINING REPORT ==========")
    print("Pairs used              :", len(pairs))
    print("Rows with negatives     :", len(mined_rows))
    print("Skipped no negative     :", skipped_no_negative)
    print("Total negative items    :", stats["total_negative_items"])
    print("Avg negatives per query :", stats["avg_negatives_per_query"])
    print("Saved output            :", output_path)
    print("Saved stats             :", stats_path)
    print("Saved samples           :", sample_path)
    print("=================================================\n")

    print("[SAMPLE HARD NEGATIVES]")
    for row in mined_rows[:3]:
        print("-" * 100)
        print("Q:", row["query"])
        print("POS:", row["positive_legal_reference_keys"], "|", row["positive_retrieval_title"])

        for j, neg in enumerate(row["negatives"][:3], start=1):
            print(f"NEG {j}:", neg["negative_legal_reference_keys"], "|", neg["negative_retrieval_title"])


if __name__ == "__main__":
    main()