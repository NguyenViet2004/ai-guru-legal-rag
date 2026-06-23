import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict


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


def save_jsonl(path: Path, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_metadata_value(metadata: dict, keys: list, default=""):
    for key in keys:
        value = metadata.get(key)
        if value not in [None, "", []]:
            return value
    return default


def get_positive_text(chunk: dict) -> str:
    return (
        chunk.get("search_text")
        or chunk.get("chunk_text")
        or chunk.get("content")
        or ""
    )


def get_article_number(metadata: dict, citation: str = "") -> str:
    article_number = get_metadata_value(
        metadata,
        ["article_number", "article_id", "dieu"],
        "",
    )

    if article_number:
        return str(article_number)

    m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", citation or "", flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return ""


def normalize_question(q: str) -> str:
    q = clean_text(q)
    q = q.replace("..", ".")
    q = q.replace("??", "?")

    if not q:
        return ""

    if not q.endswith("?"):
        q += "?"

    return q


def make_queries_from_chunk(chunk: dict, max_queries_per_chunk: int = 8) -> list:
    metadata = chunk.get("metadata", {}) or {}

    citation = chunk.get("citation", "") or ""
    retrieval_title = chunk.get("retrieval_title", "") or ""
    content = chunk.get("content", "") or ""

    document_number = get_metadata_value(
        metadata,
        ["document_number", "doc_number", "so_hieu_van_ban"],
        "",
    )

    document_title = get_metadata_value(
        metadata,
        ["document_title", "title", "van_ban"],
        "",
    )

    document_type = get_metadata_value(
        metadata,
        ["document_type", "type"],
        "",
    )

    article_title = get_metadata_value(
        metadata,
        ["article_title", "title_article", "ten_dieu"],
        "",
    )

    article_number = get_article_number(metadata, citation)

    legal_reference_keys = metadata.get("legal_reference_keys", []) or []
    citation_aliases = metadata.get("citation_aliases", []) or []
    plain_language_aliases = metadata.get("plain_language_aliases", []) or []

    chunk_level = chunk.get("chunk_level", "")

    queries = []

    # 1. Query theo Điều
    if article_number and document_title:
        queries.append(f"Điều {article_number} {document_title} quy định gì")
        queries.append(f"Nội dung Điều {article_number} của {document_title} là gì")

    if article_number and document_number:
        queries.append(f"Điều {article_number} văn bản {document_number} quy định gì")

    # 2. Query theo tiêu đề Điều
    if article_title:
        title = clean_text(article_title)

        queries.append(f"{title} được quy định như thế nào")
        queries.append(f"Quy định về {title.lower()} là gì")
        queries.append(f"Căn cứ pháp lý về {title.lower()} là gì")

        if document_type:
            queries.append(f"{document_type} quy định gì về {title.lower()}")

    # 3. Query theo alias tự nhiên nếu có
    for alias in plain_language_aliases[:3]:
        alias = clean_text(alias)
        if len(alias) >= 5:
            queries.append(f"Quy định pháp luật về {alias} là gì")
            queries.append(f"Doanh nghiệp cần lưu ý gì về {alias}")

    # 4. Query theo citation alias
    for alias in citation_aliases[:2]:
        alias = clean_text(alias)
        if len(alias) >= 5:
            queries.append(f"{alias} quy định nội dung gì")

    # 5. Query theo legal reference key
    for ref in legal_reference_keys[:2]:
        queries.append(f"{ref} quy định gì")

    # 6. Một số template theo domain phổ biến
    title_and_content = f"{retrieval_title} {content}".lower()

    if "hóa đơn điện tử" in title_and_content:
        queries.append("Doanh nghiệp sử dụng hóa đơn điện tử khi nào")
        queries.append("Quy định về hóa đơn điện tử là gì")

    if "đăng ký doanh nghiệp" in title_and_content:
        queries.append("Quy định về đăng ký doanh nghiệp là gì")
        queries.append("Thủ tục đăng ký doanh nghiệp được quy định như thế nào")

    if "người lao động" in title_and_content or "hợp đồng lao động" in title_and_content:
        queries.append("Quy định về người lao động là gì")
        queries.append("Doanh nghiệp cần lưu ý gì trong quan hệ lao động")

    if "dữ liệu cá nhân" in title_and_content:
        queries.append("Quy định về xử lý dữ liệu cá nhân là gì")
        queries.append("Doanh nghiệp cần tuân thủ gì khi xử lý dữ liệu cá nhân")

    if "đất đai" in title_and_content:
        queries.append("Quy định pháp luật về đất đai là gì")

    if "môi trường" in title_and_content:
        queries.append("Doanh nghiệp cần tuân thủ quy định môi trường nào")

    if "đấu thầu" in title_and_content:
        queries.append("Quy định về đấu thầu là gì")

    # Loại trùng và query quá ngắn
    normalized = []
    seen = set()

    for q in queries:
        q = normalize_question(q)

        if len(q) < 15:
            continue

        key = q.lower()

        if key not in seen:
            seen.add(key)
            normalized.append(q)

    return normalized[:max_queries_per_chunk]


def should_use_chunk(chunk: dict, min_content_len: int) -> bool:
    content = chunk.get("content", "") or ""
    search_text = get_positive_text(chunk)

    if len(content.strip()) < min_content_len:
        return False

    if len(search_text.strip()) < min_content_len:
        return False

    # Các chunk quá đặc thù nhưng vẫn có thể có giá trị; không loại amended.
    return True


def split_train_dev(rows: list, dev_ratio: float, seed: int):
    random.seed(seed)

    rows = rows[:]
    random.shuffle(rows)

    dev_size = int(len(rows) * dev_ratio)

    dev_rows = rows[:dev_size]
    train_rows = rows[dev_size:]

    return train_rows, dev_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/kaggle/input/datasets/nguyenviet2709/legal-v3/chunk_v3",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
    )

    parser.add_argument(
        "--chunk_file_name",
        type=str,
        default="legal_chunks_final.jsonl",
    )

    parser.add_argument(
        "--max_queries_per_chunk",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min_content_len",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    rows = []
    skipped_short = 0
    no_query = 0

    chunk_level_counter = Counter()
    query_count_by_chunk_level = Counter()
    document_type_counter = Counter()
    samples_by_level = defaultdict(list)

    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        chunk_level = chunk.get("chunk_level", "UNKNOWN")
        document_type = metadata.get("document_type", "UNKNOWN")

        chunk_level_counter[chunk_level] += 1
        document_type_counter[document_type] += 1

        if not should_use_chunk(chunk, args.min_content_len):
            skipped_short += 1
            continue

        queries = make_queries_from_chunk(
            chunk,
            max_queries_per_chunk=args.max_queries_per_chunk,
        )

        if not queries:
            no_query += 1
            continue

        positive_text = get_positive_text(chunk)

        for q in queries:
            item = {
                "query": q,
                "positive_chunk_id": chunk.get("chunk_id"),
                "parent_article_id": chunk.get("parent_article_id"),
                "chunk_level": chunk.get("chunk_level"),
                "positive_text": positive_text,
                "positive_content": chunk.get("content", ""),
                "citation": chunk.get("citation"),
                "retrieval_title": chunk.get("retrieval_title"),
                "legal_reference_keys": metadata.get("legal_reference_keys", []) or [],
                "document_number": metadata.get("document_number"),
                "document_title": metadata.get("document_title"),
                "document_type": metadata.get("document_type"),
                "source": "template_synthetic_v1",
            }

            rows.append(item)
            query_count_by_chunk_level[chunk_level] += 1

            if len(samples_by_level[chunk_level]) < 3:
                samples_by_level[chunk_level].append({
                    "query": q,
                    "ref": metadata.get("legal_reference_keys", []),
                    "title": chunk.get("retrieval_title"),
                })

    # Loại trùng theo query + positive_chunk_id
    deduped = []
    seen = set()

    for row in rows:
        key = (row["query"].lower(), row["positive_chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    train_rows, dev_rows = split_train_dev(
        deduped,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    train_path = artifact_dir / "synthetic_train_pairs.jsonl"
    dev_path = artifact_dir / "synthetic_dev_pairs.jsonl"
    all_path = artifact_dir / "synthetic_all_pairs.jsonl"
    sample_path = artifact_dir / "synthetic_samples.json"
    stats_path = artifact_dir / "synthetic_stats.json"

    save_jsonl(train_path, train_rows)
    save_jsonl(dev_path, dev_rows)
    save_jsonl(all_path, deduped)

    stats = {
        "chunk_path": str(chunk_path),
        "total_chunks": len(chunks),
        "used_chunks_estimate": len(set(row["positive_chunk_id"] for row in deduped)),
        "skipped_short": skipped_short,
        "no_query": no_query,
        "total_pairs_before_dedup": len(rows),
        "total_pairs_after_dedup": len(deduped),
        "train_pairs": len(train_rows),
        "dev_pairs": len(dev_rows),
        "max_queries_per_chunk": args.max_queries_per_chunk,
        "min_content_len": args.min_content_len,
        "dev_ratio": args.dev_ratio,
        "chunk_level_distribution": dict(chunk_level_counter),
        "query_count_by_chunk_level": dict(query_count_by_chunk_level),
        "document_type_distribution": dict(document_type_counter),
        "output_files": {
            "train": str(train_path),
            "dev": str(dev_path),
            "all": str(all_path),
            "samples": str(sample_path),
        },
    }

    save_json(stats_path, stats)
    save_json(sample_path, dict(samples_by_level))

    print("\n========== SYNTHETIC DATA REPORT ==========")
    print("Total chunks             :", len(chunks))
    print("Skipped short chunks     :", skipped_short)
    print("Chunks without query     :", no_query)
    print("Pairs before dedup       :", len(rows))
    print("Pairs after dedup        :", len(deduped))
    print("Train pairs              :", len(train_rows))
    print("Dev pairs                :", len(dev_rows))
    print("Saved train              :", train_path)
    print("Saved dev                :", dev_path)
    print("Saved stats              :", stats_path)
    print("===========================================\n")

    print("[SAMPLE]")
    for sample in deduped[:10]:
        print("-" * 100)
        print("Q   :", sample["query"])
        print("Ref :", sample["legal_reference_keys"])
        print("Text:", sample["positive_content"][:250])


if __name__ == "__main__":
    main()