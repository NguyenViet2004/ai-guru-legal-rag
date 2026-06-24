import argparse
import gc
import importlib.util
import json
from pathlib import Path

import faiss
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def clean_text(text: str, max_chars: int = 5000) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text[:max_chars]


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


def rerank_with_loaded_model(
    query: str,
    results: list[dict],
    lookup: dict,
    reranker,
    max_chars: int,
):
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
    seen_refs = set()
    seen_title = set()

    for result in results:
        parent_id = result.get("parent_article_id")
        refs = tuple(result.get("legal_reference_keys", []) or [])
        title = result.get("retrieval_title")

        if parent_id and parent_id in seen_parent:
            continue

        if refs and refs in seen_refs:
            continue

        if title and title in seen_title:
            continue

        if parent_id:
            seen_parent.add(parent_id)

        if refs:
            seen_refs.add(refs)

        if title:
            seen_title.add(title)

        selected.append(result)

        if len(selected) >= top_k:
            break

    return selected


def is_broad_question(question: str) -> bool:
    q = question.lower()

    broad_signals = [
        "là gì",
        "gồm những gì",
        "bao gồm những gì",
        "quy định về",
        "trách nhiệm",
        "nghĩa vụ",
        "điều kiện",
        "nguyên tắc",
        "các trường hợp",
    ]

    return any(signal in q for signal in broad_signals)


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
- Chỉ trả lời dựa trên các căn cứ pháp lý được cung cấp.
- Không được bịa nội dung ngoài căn cứ.
- Nếu căn cứ chưa đủ để kết luận, hãy nói rõ là chưa đủ căn cứ.
- Trả lời ngắn gọn, rõ ràng, đúng trọng tâm.
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


def build_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý pháp lý cho doanh nghiệp tại Việt Nam. "
                "Chỉ trả lời dựa trên căn cứ pháp lý được cung cấp. "
                "Không bịa nội dung ngoài căn cứ."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def load_llm(model_name: str, load_in_4bit: bool):
    print("[INFO] Loading tokenizer:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading LLM:", model_name)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def generate_answer_with_loaded_model(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    messages = build_messages(prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=12000,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer


def extract_legal_refs_used(answer: str, contexts: list[dict]) -> list[str]:
    used_refs = []
    answer_lower = answer.lower()

    for ctx in contexts:
        refs = ctx.get("legal_reference_keys", []) or []
        citation = ctx.get("citation", "") or ""

        for ref in refs:
            ref_lower = ref.lower()

            if ref_lower in answer_lower and ref not in used_refs:
                used_refs.append(ref)
                continue

            parts = ref.split("|")

            if len(parts) >= 2:
                doc_no = parts[0].lower()
                article = parts[1].lower()

                if doc_no in answer_lower and article in answer_lower and ref not in used_refs:
                    used_refs.append(ref)
                    continue

            if citation and citation.lower() in answer_lower and ref not in used_refs:
                used_refs.append(ref)

    if used_refs:
        return used_refs

    fallback_refs = []

    for ctx in contexts[:3]:
        for ref in ctx.get("legal_reference_keys", []) or []:
            if ref not in fallback_refs:
                fallback_refs.append(ref)

    return fallback_refs


def matches_expected(legal_refs: list[str], expected_refs: list[str]) -> bool:
    combined = " ".join(legal_refs)

    for ref in expected_refs:
        if ref in combined:
            return True

    return False


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
        "--reranker_model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
    )

    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    parser.add_argument(
        "--load_llm_in_4bit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--use_reranker",
        type=int,
        default=1,
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
        "--specific_context_top_k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--broad_context_top_k",
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
        "--max_new_tokens",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="/kaggle/working/artifacts/manual_qa_outputs.json",
    )

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)

    eval_file = Path(args.eval_file)

    if not eval_file.is_absolute():
        eval_file = root_dir / eval_file

    output_file = Path(args.output_file)

    hybrid_module_path = root_dir / "scripts" / "04_hybrid_retrieval.py"
    hybrid = load_module_from_path("hybrid_retrieval", hybrid_module_path)

    print("[INFO] Eval file:", eval_file)
    print("[INFO] Artifact dir:", artifact_dir)

    eval_data = load_yaml(eval_file)
    eval_items = eval_data["queries"]

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    print("[INFO] Loading BM25:", bm25_path)
    bm25 = hybrid.load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = hybrid.load_pickle(chunks_path)

    print("[INFO] Building lookup")
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

    reranker = None

    if args.use_reranker:
        print("[INFO] Loading reranker:", args.reranker_model_name)
        reranker = CrossEncoder(
            args.reranker_model_name,
            max_length=512,
            device=device,
        )

    prepared_items = []

    print("\n========== PHASE 1: PREPARE CONTEXTS ==========")

    for item in eval_items:
        qid = item["id"]
        question = item["question"]
        expected_any = item.get("expected_any", [])

        context_top_k = (
            args.broad_context_top_k
            if is_broad_question(question)
            else args.specific_context_top_k
        )

        print("-" * 100)
        print("ID:", qid)
        print("Q :", question)
        print("Context top k:", context_top_k)

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

        results = [dict(r) for r in hybrid_results]

        if reranker is not None:
            results = rerank_with_loaded_model(
                query=question,
                results=results,
                lookup=lookup,
                reranker=reranker,
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
            top_k=context_top_k,
        )

        context_blocks = build_context_blocks(
            selected_results=selected,
            lookup=lookup,
            max_context_chars=args.max_context_chars,
        )

        prompt = build_prompt(
            question=question,
            context_blocks=context_blocks,
        )

        print("Top context:", context_blocks[0]["legal_reference_keys"], "|", context_blocks[0]["retrieval_title"])

        prepared_items.append({
            "id": qid,
            "question": question,
            "expected_any": expected_any,
            "context_top_k": context_top_k,
            "contexts": context_blocks,
            "prompt": prompt,
        })

    # Giải phóng retrieval model trước khi load LLM
    print("\n[INFO] Releasing retrieval models before loading LLM")

    del dense_model
    del dense_index
    del bm25
    del chunks

    if reranker is not None:
        del reranker

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== PHASE 2: GENERATE ANSWERS ==========")

    tokenizer, llm = load_llm(
        model_name=args.llm_model_name,
        load_in_4bit=bool(args.load_llm_in_4bit),
    )

    outputs = []
    matched_count = 0

    for item in prepared_items:
        qid = item["id"]
        question = item["question"]

        print("-" * 100)
        print("ID:", qid)
        print("Q :", question)

        answer = generate_answer_with_loaded_model(
            prompt=item["prompt"],
            tokenizer=tokenizer,
            model=llm,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        legal_refs = extract_legal_refs_used(
            answer=answer,
            contexts=item["contexts"],
        )

        expected_hit = matches_expected(
            legal_refs=legal_refs,
            expected_refs=item["expected_any"],
        )

        if expected_hit:
            matched_count += 1

        out = {
            "id": qid,
            "question": question,
            "answer": answer,
            "legal_refs": legal_refs,
            "expected_any": item["expected_any"],
            "expected_ref_hit": expected_hit,
            "context_top_k": item["context_top_k"],
            "contexts": item["contexts"],
        }

        outputs.append(out)

        print("Expected:", item["expected_any"])
        print("Legal refs:", legal_refs)
        print("Expected ref hit:", expected_hit)
        print("Answer preview:", answer[:500])

    summary = {
        "total": len(outputs),
        "expected_ref_hit_count": matched_count,
        "expected_ref_hit_rate": matched_count / len(outputs) if outputs else 0,
        "embedding_model_name": args.embedding_model_name,
        "reranker_model_name": args.reranker_model_name if args.use_reranker else None,
        "llm_model_name": args.llm_model_name,
        "use_reranker": bool(args.use_reranker),
        "hybrid_weight": args.hybrid_weight,
        "reranker_weight": args.reranker_weight if args.use_reranker else None,
        "specific_context_top_k": args.specific_context_top_k,
        "broad_context_top_k": args.broad_context_top_k,
    }

    report = {
        "summary": summary,
        "outputs": outputs,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_file, report)

    print("\n========== MANUAL QA REPORT ==========")
    print("Total:", summary["total"])
    print("Expected ref hit:", summary["expected_ref_hit_count"], "/", summary["total"])
    print("Expected ref hit rate:", summary["expected_ref_hit_rate"])
    print("Saved:", output_file)
    print("======================================\n")


if __name__ == "__main__":
    main()