import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_legal_refs_used(answer: str, contexts: list[dict]) -> list[str]:
    """
    Ưu tiên lấy căn cứ thật sự xuất hiện trong câu trả lời.
    Nếu không bắt được thì fallback lấy context top 1-3.
    """
    used_refs = []

    answer_lower = answer.lower()

    for ctx in contexts:
        refs = ctx.get("legal_reference_keys", []) or []
        title = ctx.get("retrieval_title", "") or ""
        citation = ctx.get("citation", "") or ""

        searchable = " ".join(refs + [title, citation]).lower()

        for ref in refs:
            ref_lower = ref.lower()

            # Bắt theo mã ref đầy đủ
            if ref_lower in answer_lower and ref not in used_refs:
                used_refs.append(ref)

            # Bắt theo Điều + số hiệu văn bản
            parts = ref.split("|")
            if len(parts) >= 2:
                doc_no = parts[0].lower()
                article = parts[1].lower()

                if doc_no in answer_lower and article in answer_lower and ref not in used_refs:
                    used_refs.append(ref)

            # Bắt nếu câu trả lời nhắc Điều trong title/citation
            if ref not in used_refs:
                if citation and citation.lower() in answer_lower:
                    used_refs.append(ref)

    if used_refs:
        return used_refs

    # Fallback: lấy tối đa 3 căn cứ đầu nếu không parse được
    fallback_refs = []

    for ctx in contexts[:3]:
        for ref in ctx.get("legal_reference_keys", []) or []:
            if ref not in fallback_refs:
                fallback_refs.append(ref)

    return fallback_refs


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


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool):
    print("[INFO] Loading tokenizer:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model:", model_name)

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


def generate_answer(
    prompt: str,
    model_name: str,
    load_in_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

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

    print("[INFO] Device:", device)
    print("[INFO] Input tokens:", inputs["input_ids"].shape[-1])

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    parser.add_argument(
        "--load_in_4bit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
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

    args = parser.parse_args()

    context_path = Path(args.context_file)
    output_path = Path(args.output_file)

    if not context_path.exists():
        raise FileNotFoundError(f"Không tìm thấy context file: {context_path}")

    rag_data = load_json(context_path)

    question = rag_data.get("question", "")
    prompt = rag_data.get("prompt", "")
    contexts = rag_data.get("contexts", [])

    if not prompt:
        raise ValueError("File context không có trường prompt.")

    print("[INFO] Context file:", context_path)
    print("[INFO] Question:", question)
    print("[INFO] Context count:", len(contexts))
    print("[INFO] Model:", args.model_name)
    print("[INFO] Load in 4bit:", bool(args.load_in_4bit))

    answer = generate_answer(
        prompt=prompt,
        model_name=args.model_name,
        load_in_4bit=bool(args.load_in_4bit),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    legal_refs = extract_legal_refs_used(answer, contexts)

    output = {
        "question": question,
        "answer": answer,
        "legal_refs": legal_refs,
        "model_name": args.model_name,
        "load_in_4bit": bool(args.load_in_4bit),
        "context_file": str(context_path),
        "contexts": contexts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output)

    print("\n========== GENERATED ANSWER ==========")
    print("Question:", question)
    print("-" * 100)
    print(answer)
    print("-" * 100)
    print("Legal refs:", legal_refs)
    print("Saved:", output_path)
    print("======================================\n")


if __name__ == "__main__":
    main()