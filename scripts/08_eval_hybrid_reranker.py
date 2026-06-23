import argparse
import importlib.util
import json
from pathlib import Path

import faiss
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str, max_chars: int = 3500) -> str:
    if not text:
        return ""

    text = " ".join(text.split())
    return text[:max_chars]


def build_passage(result: dict, max_chars: int = 3500) -> str:
    title = result.get("retrieval_title") or ""
    citation = result.get("citation") or ""
    content = result.get("content_preview") or ""

    parts = []

    if title:
        parts.append(f"Tiêu đề: {title}")

    if citation:
        parts.append(f"Căn cứ: {citation}")

    if content:
        parts.append(f"Nội dung: {clean_text(content, max_chars=max_chars)}")

    return "\n".join(parts)


def result_matches_expected(result: dict, expected_refs: list) -> bool:
    legal_keys = result.get("legal_reference_keys", []) or []
    citation = result.get("citation", "") or ""
    title = result.get("retrieval_title", "") or ""

    combined = " ".join(legal_keys + [citation, title])

    for ref in expected_refs:
        if ref in combined:
            return True

    return False


def evaluate_one(results: list, expected_refs: list) -> dict:
    hit_at = {}

    for k in [1, 3, 5, 10]:
        top_k = results[:k]
        hit_at[f"hit@{k}"] = any(
            result_matches_expected(r, expected_refs)
            for r in top_k
        )

    matched_rank = None

    for r in results:
        if result_matches_expected(r, expected_refs):
            matched_rank = r["rank"]
            break

    return {
        "matched_rank": matched_rank,
        **hit_at,
    }


def rerank_results(query: str, results: list, reranker, max_chars: int = 3500):
    pairs = []

    for r in results:
        passage = build_passage(r, max_chars=max_chars)
        pairs.append([query, passage])

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    reranked = []

    for r, score in zip(results, scores):
        item = dict(r)
        item["reranker_score"] = float(score)
        reranked.append(item)

    reranked = sorted(
        reranked,
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    for i, r in enumerate(reranked, start=1):
        r["rank"] = i

    return reranked


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
    )

    parser.add_argument(
        "--eval_file",
        type=str,
        default="configs/manual_eval_queries.yaml",
    )

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="BAAI/bge-m3",
    )

    parser.add_argument(
        "--reranker_model_dir",
        type=str,
        default="/kaggle/working/artifacts/reranker_bge_m3_ft_test",
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
        "--final_top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_chars",
        type=int,
        default=3500,
    )

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)

    eval_file = Path(args.eval_file)

    if not eval_file.is_absolute():
        eval_file = root_dir / eval_file

    hybrid_module_path = root_dir / "scripts" / "04_hybrid_retrieval.py"
    hybrid = load_module_from_path("hybrid_retrieval", hybrid_module_path)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    print("[INFO] Eval file:", eval_file)
    print("[INFO] Reranker :", args.reranker_model_dir)

    eval_data = load_yaml(eval_file)
    queries = eval_data["queries"]

    print("[INFO] Loading BM25:", bm25_path)
    bm25 = hybrid.load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = hybrid.load_pickle(chunks_path)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Device:", device)

    print("[INFO] Loading embedding model:", args.embedding_model_name)
    dense_model = SentenceTransformer(args.embedding_model_name, device=device)

    print("[INFO] Loading reranker:", args.reranker_model_dir)
    reranker = CrossEncoder(
        args.reranker_model_dir,
        max_length=512,
        device=device,
    )

    detailed_results = []

    metric_counts = {
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
        "hit@10": 0,
    }

    for item in queries:
        qid = item["id"]
        question = item["question"]
        expected_any = item.get("expected_any", [])

        hybrid_results = hybrid.hybrid_search(
            query=question,
            chunks=chunks,
            bm25=bm25,
            dense_model=dense_model,
            dense_index=dense_index,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            final_top_k=args.candidate_top_k,
        )

        reranked_results = rerank_results(
            query=question,
            results=hybrid_results,
            reranker=reranker,
            max_chars=args.max_chars,
        )[:args.final_top_k]

        eval_result = evaluate_one(reranked_results, expected_any)

        for k in metric_counts:
            if eval_result[k]:
                metric_counts[k] += 1

        top_1 = reranked_results[0] if reranked_results else {}

        detailed_results.append({
            "id": qid,
            "question": question,
            "expected_any": expected_any,
            "matched_rank": eval_result["matched_rank"],
            "hit@1": eval_result["hit@1"],
            "hit@3": eval_result["hit@3"],
            "hit@5": eval_result["hit@5"],
            "hit@10": eval_result["hit@10"],
            "top1_ref": top_1.get("legal_reference_keys", []),
            "top1_title": top_1.get("retrieval_title"),
            "top1_reranker_score": top_1.get("reranker_score"),
            "top_results": reranked_results,
        })

        print("-" * 120)
        print("ID:", qid)
        print("Q :", question)
        print("Expected:", expected_any)
        print("Matched rank:", eval_result["matched_rank"])
        print("Hit@1/3/5/10:", eval_result["hit@1"], eval_result["hit@3"], eval_result["hit@5"], eval_result["hit@10"])
        print("Top1 score:", top_1.get("reranker_score"))
        print("Top1:", top_1.get("legal_reference_keys"), "|", top_1.get("retrieval_title"))

    total = len(queries)

    summary = {
        "total": total,
        "hit@1": metric_counts["hit@1"] / total,
        "hit@3": metric_counts["hit@3"] / total,
        "hit@5": metric_counts["hit@5"] / total,
        "hit@10": metric_counts["hit@10"] / total,
        "counts": metric_counts,
    }

    report = {
        "summary": summary,
        "details": detailed_results,
    }

    out_path = artifact_dir / "manual_eval_reranker_report.json"
    save_json(out_path, report)

    print("\n========== HYBRID + RERANKER EVAL SUMMARY ==========")
    print("Total:", total)
    print("Hit@1 :", summary["hit@1"], metric_counts["hit@1"], "/", total)
    print("Hit@3 :", summary["hit@3"], metric_counts["hit@3"], "/", total)
    print("Hit@5 :", summary["hit@5"], metric_counts["hit@5"], "/", total)
    print("Hit@10:", summary["hit@10"], metric_counts["hit@10"], "/", total)
    print("Saved:", out_path)
    print("====================================================\n")


if __name__ == "__main__":
    main()