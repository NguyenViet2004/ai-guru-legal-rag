import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict


BAD_QUERY_PATTERNS = [
    r"^Điều\s+\d+[a-zA-Z]?\s+Về\s+",
    r"^Nội dung Điều\s+\d+[a-zA-Z]?\s+của\s+Về\s+",
    r"^(NGHỊ ĐỊNH|LUẬT|BỘ LUẬT|THÔNG TƯ)\s+quy định gì về",
]

GENERIC_PHRASES = [
    "phạm vi điều chỉnh",
    "đối tượng áp dụng",
    "hiệu lực thi hành",
    "trách nhiệm thi hành",
    "tổ chức thực hiện",
    "điều khoản thi hành",
]

TOO_BROAD_QUERIES = [
    "quy định về đăng ký doanh nghiệp là gì",
    "thủ tục đăng ký doanh nghiệp được quy định như thế nào",
    "quy định về người lao động là gì",
    "doanh nghiệp cần lưu ý gì trong quan hệ lao động",
    "quy định về hóa đơn điện tử là gì",
    "quy định về xử lý dữ liệu cá nhân là gì",
    "doanh nghiệp cần tuân thủ gì khi xử lý dữ liệu cá nhân",
    "quy định pháp luật về đất đai là gì",
    "doanh nghiệp cần tuân thủ quy định môi trường nào",
    "quy định về đấu thầu là gì",
]


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


def normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = q.rstrip("?!. ")
    return q


def is_bad_pattern(query: str) -> bool:
    for pattern in BAD_QUERY_PATTERNS:
        if re.search(pattern, query):
            return True

    return False


def is_generic_query(query: str) -> bool:
    q = normalize_query(query)

    for phrase in GENERIC_PHRASES:
        if phrase in q:
            return True

    for broad in TOO_BROAD_QUERIES:
        if q == broad:
            return True

    return False


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
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="synthetic_all_pairs.jsonl",
    )

    parser.add_argument(
        "--max_positive_per_query",
        type=int,
        default=1,
        help="Chỉ giữ query xuất hiện với tối đa N positive_chunk_id. Để train MNRL nên dùng 1.",
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

    artifact_dir = Path(args.artifact_dir)
    input_path = artifact_dir / args.input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {input_path}. Hãy chạy 05_generate_synthetic_data.py trước.")

    rows = load_jsonl(input_path)

    print("[INFO] Loaded rows:", len(rows))

    # Đếm số positive_chunk_id khác nhau cho mỗi query
    query_to_chunk_ids = defaultdict(set)

    for row in rows:
        q_norm = normalize_query(row["query"])
        query_to_chunk_ids[q_norm].add(row["positive_chunk_id"])

    query_positive_count = {
        q: len(chunk_ids)
        for q, chunk_ids in query_to_chunk_ids.items()
    }

    filtered = []
    drop_reasons = Counter()

    seen_pair = set()

    for row in rows:
        query = row["query"]
        q_norm = normalize_query(query)

        if is_bad_pattern(query):
            drop_reasons["bad_pattern"] += 1
            continue

        if is_generic_query(query):
            drop_reasons["generic_or_too_broad"] += 1
            continue

        if query_positive_count[q_norm] > args.max_positive_per_query:
            drop_reasons["query_has_many_positives"] += 1
            continue

        if len(q_norm) < 15:
            drop_reasons["too_short"] += 1
            continue

        key = (q_norm, row["positive_chunk_id"])

        if key in seen_pair:
            drop_reasons["duplicate_pair"] += 1
            continue

        seen_pair.add(key)
        filtered.append(row)

    train_rows, dev_rows = split_train_dev(
        filtered,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    train_path = artifact_dir / "synthetic_train_pairs_v2.jsonl"
    dev_path = artifact_dir / "synthetic_dev_pairs_v2.jsonl"
    all_path = artifact_dir / "synthetic_all_pairs_v2.jsonl"
    stats_path = artifact_dir / "synthetic_filter_stats_v2.json"
    sample_path = artifact_dir / "synthetic_samples_v2.json"

    save_jsonl(train_path, train_rows)
    save_jsonl(dev_path, dev_rows)
    save_jsonl(all_path, filtered)

    samples = filtered[:30]

    stats = {
        "input_file": str(input_path),
        "total_input_pairs": len(rows),
        "total_filtered_pairs": len(filtered),
        "train_pairs": len(train_rows),
        "dev_pairs": len(dev_rows),
        "drop_reasons": dict(drop_reasons),
        "max_positive_per_query": args.max_positive_per_query,
        "unique_queries_before": len(query_to_chunk_ids),
        "unique_queries_after": len(set(normalize_query(r["query"]) for r in filtered)),
        "output_files": {
            "train": str(train_path),
            "dev": str(dev_path),
            "all": str(all_path),
            "stats": str(stats_path),
            "samples": str(sample_path),
        },
    }

    save_json(stats_path, stats)
    save_json(sample_path, samples)

    print("\n========== SYNTHETIC FILTER REPORT ==========")
    print("Input pairs          :", len(rows))
    print("Filtered pairs       :", len(filtered))
    print("Train pairs v2       :", len(train_rows))
    print("Dev pairs v2         :", len(dev_rows))
    print("Unique queries before:", stats["unique_queries_before"])
    print("Unique queries after :", stats["unique_queries_after"])
    print("Drop reasons         :", dict(drop_reasons))
    print("Saved train          :", train_path)
    print("Saved dev            :", dev_path)
    print("Saved stats          :", stats_path)
    print("=============================================\n")

    print("[SAMPLE V2]")
    for sample in samples[:10]:
        print("-" * 100)
        print("Q   :", sample["query"])
        print("Ref :", sample["legal_reference_keys"])
        print("Title:", sample["retrieval_title"])


if __name__ == "__main__":
    main()