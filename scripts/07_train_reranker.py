import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator


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


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str, max_chars: int = 3500) -> str:
    if not text:
        return ""

    text = " ".join(text.split())
    return text[:max_chars]


def build_passage(title: str, citation: str, content: str) -> str:
    parts = []

    if title:
        parts.append(f"Tiêu đề: {title}")

    if citation:
        parts.append(f"Căn cứ: {citation}")

    if content:
        parts.append(f"Nội dung: {content}")

    return "\n".join(parts)


def make_examples(rows: list, negatives_per_query: int, max_chars: int):
    examples = []

    for row in rows:
        query = clean_text(row.get("query", ""), max_chars=800)

        if not query:
            continue

        positive_passage = build_passage(
            title=row.get("positive_retrieval_title", ""),
            citation=row.get("positive_citation", ""),
            content=clean_text(
                row.get("positive_content") or row.get("positive_text") or "",
                max_chars=max_chars,
            ),
        )

        if positive_passage:
            examples.append(
                InputExample(
                    texts=[query, positive_passage],
                    label=1.0,
                )
            )

        negatives = row.get("negatives", [])[:negatives_per_query]

        for neg in negatives:
            negative_passage = build_passage(
                title=neg.get("negative_retrieval_title", ""),
                citation=neg.get("negative_citation", ""),
                content=clean_text(
                    neg.get("negative_content") or neg.get("negative_text") or "",
                    max_chars=max_chars,
                ),
            )

            if negative_passage:
                examples.append(
                    InputExample(
                        texts=[query, negative_passage],
                        label=0.0,
                    )
                )

    return examples


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
        default="hard_negative_train_v1.jsonl",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/artifacts/reranker_bge_m3_ft",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=5000,
        help="Số query rows dùng train thử. 0 nghĩa là dùng toàn bộ.",
    )

    parser.add_argument(
        "--negatives_per_query",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max_chars",
        type=int,
        default=3500,
    )

    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    input_path = artifact_dir / args.input_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {input_path}")

    print("[INFO] Input:", input_path)
    print("[INFO] Output dir:", output_dir)
    print("[INFO] Model:", args.model_name)

    rows = load_jsonl(input_path)

    random.shuffle(rows)

    if args.max_rows and args.max_rows > 0:
        rows = rows[:args.max_rows]

    dev_size = max(1, int(len(rows) * args.dev_ratio))

    dev_rows = rows[:dev_size]
    train_rows = rows[dev_size:]

    print("[INFO] Rows used:", len(rows))
    print("[INFO] Train rows:", len(train_rows))
    print("[INFO] Dev rows:", len(dev_rows))

    train_examples = make_examples(
        train_rows,
        negatives_per_query=args.negatives_per_query,
        max_chars=args.max_chars,
    )

    dev_examples = make_examples(
        dev_rows,
        negatives_per_query=args.negatives_per_query,
        max_chars=args.max_chars,
    )

    print("[INFO] Train examples:", len(train_examples))
    print("[INFO] Dev examples:", len(dev_examples))

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] Device:", device)

    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_examples,
        name="legal-reranker-dev",
    )

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

    print("[INFO] Warmup steps:", warmup_steps)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": args.learning_rate},
        use_amp=True,
    )

    report = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "rows_used": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "negatives_per_query": args.negatives_per_query,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_chars": args.max_chars,
        "device": device,
    }

    save_json(output_dir / "train_report.json", report)

    print("\n========== RERANKER TRAIN REPORT ==========")
    print("Rows used      :", len(rows))
    print("Train examples :", len(train_examples))
    print("Dev examples   :", len(dev_examples))
    print("Output dir     :", output_dir)
    print("===========================================\n")


if __name__ == "__main__":
    main()