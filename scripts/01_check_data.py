import argparse
import json
import zipfile
from pathlib import Path
from collections import Counter


REQUIRED_TOP_LEVEL_FIELDS = [
    "chunk_id",
    "parent_article_id",
    "chunk_level",
    "citation",
    "retrieval_title",
    "content",
    "chunk_text",
    "search_text",
    "metadata",
    "retrieval",
]


REQUIRED_METADATA_FIELDS = [
    "document_number",
    "document_title",
    "document_type",
    "field_group",
    "legal_reference_keys",
    "citation_aliases",
]


def find_zip_file(input_dir: Path) -> Path:
    zip_files = list(input_dir.rglob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(f"Không tìm thấy file .zip trong {input_dir}")

    if len(zip_files) > 1:
        print("[WARNING] Tìm thấy nhiều file zip:")
        for z in zip_files:
            print(" -", z)
        print("[INFO] Sẽ dùng file zip đầu tiên:", zip_files[0])

    return zip_files[0]


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting: {zip_path}")
    print(f"[INFO] To        : {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


def find_chunk_file(extract_dir: Path, chunk_file_name: str) -> Path:
    candidates = list(extract_dir.rglob(chunk_file_name))

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy {chunk_file_name} trong {extract_dir}"
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


def validate_chunks(chunks: list) -> dict:
    report = {}

    report["total_chunks"] = len(chunks)

    chunk_ids = [c.get("chunk_id") for c in chunks]
    parent_ids = [c.get("parent_article_id") for c in chunks]

    report["unique_chunk_ids"] = len(set(chunk_ids))
    report["duplicated_chunk_ids"] = len(chunk_ids) - len(set(chunk_ids))
    report["unique_parent_article_ids"] = len(set(parent_ids))

    missing_top_fields = Counter()
    missing_metadata_fields = Counter()

    chunk_level_counter = Counter()
    field_group_counter = Counter()
    document_type_counter = Counter()

    empty_search_text = 0
    empty_content = 0

    for c in chunks:
        for field in REQUIRED_TOP_LEVEL_FIELDS:
            if field not in c or c.get(field) in [None, ""]:
                missing_top_fields[field] += 1

        metadata = c.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        for field in REQUIRED_METADATA_FIELDS:
            if field not in metadata or metadata.get(field) in [None, "", []]:
                missing_metadata_fields[field] += 1

        chunk_level_counter[c.get("chunk_level", "UNKNOWN")] += 1
        field_group_counter[metadata.get("field_group", "UNKNOWN")] += 1
        document_type_counter[metadata.get("document_type", "UNKNOWN")] += 1

        if not c.get("search_text"):
            empty_search_text += 1

        if not c.get("content"):
            empty_content += 1

    report["missing_top_level_fields"] = dict(missing_top_fields)
    report["missing_metadata_fields"] = dict(missing_metadata_fields)
    report["chunk_level_distribution"] = dict(chunk_level_counter)
    report["field_group_count"] = len(field_group_counter)
    report["document_type_distribution"] = dict(document_type_counter)
    report["empty_search_text"] = empty_search_text
    report["empty_content"] = empty_content

    return report


def print_report(report: dict) -> None:
    print("\n========== DATA CHECK REPORT ==========")
    print("Total chunks              :", report["total_chunks"])
    print("Unique chunk IDs          :", report["unique_chunk_ids"])
    print("Duplicated chunk IDs      :", report["duplicated_chunk_ids"])
    print("Unique parent article IDs :", report["unique_parent_article_ids"])
    print("Field group count         :", report["field_group_count"])
    print("Empty search_text         :", report["empty_search_text"])
    print("Empty content             :", report["empty_content"])

    print("\nChunk level distribution:")
    for k, v in report["chunk_level_distribution"].items():
        print(f"  {k}: {v}")

    print("\nDocument type distribution:")
    for k, v in report["document_type_distribution"].items():
        print(f"  {k}: {v}")

    print("\nMissing top-level fields:")
    if report["missing_top_level_fields"]:
        for k, v in report["missing_top_level_fields"].items():
            print(f"  {k}: {v}")
    else:
        print("  None")

    print("\nMissing metadata fields:")
    if report["missing_metadata_fields"]:
        for k, v in report["missing_metadata_fields"].items():
            print(f"  {k}: {v}")
    else:
        print("  None")

    print("=======================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/kaggle/input/ai-guru-legal-chunk-v3",
        help="Thư mục Kaggle input chứa file zip chunk_v3",
    )

    parser.add_argument(
        "--extract_dir",
        type=str,
        default="/kaggle/working/chunk_v3_data",
        help="Thư mục giải nén dataset",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
        help="Thư mục lưu report",
    )

    parser.add_argument(
        "--chunk_file_name",
        type=str,
        default="legal_chunks_final.jsonl",
        help="Tên file chunk chính",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    extract_dir = Path(args.extract_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Input dir    :", input_dir)
    print("[INFO] Extract dir  :", extract_dir)
    print("[INFO] Artifact dir :", artifact_dir)

    zip_path = find_zip_file(input_dir)
    extract_zip(zip_path, extract_dir)

    chunk_path = find_chunk_file(extract_dir, args.chunk_file_name)
    print("[INFO] Chunk path:", chunk_path)

    chunks = load_jsonl(chunk_path)
    report = validate_chunks(chunks)

    report["chunk_path"] = str(chunk_path)
    report["zip_path"] = str(zip_path)

    print_report(report)

    report_path = artifact_dir / "data_check_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved report to:", report_path)


if __name__ == "__main__":
    main()