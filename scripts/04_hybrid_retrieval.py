import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list:
    text = normalize_text(text)

    tokens = re.findall(
        r"\d+/\d{4}/[a-zà-ỹđ\-]+|[a-zà-ỹđ]+|\d+",
        text,
        flags=re.IGNORECASE,
    )

    return tokens


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_chunk_text(chunk: dict) -> str:
    return (
        chunk.get("search_text")
        or chunk.get("chunk_text")
        or chunk.get("content")
        or ""
    )

def expand_query(query: str) -> str:
    """
    Mở rộng query theo intent pháp lý tổng quát.
    Chỉ dùng cho retrieval, không thay đổi câu hỏi gốc.
    """
    q = query.lower()
    expansions = []

    def has_any(*terms):
        return any(term in q for term in terms)

    def has_all(*terms):
        return all(term in q for term in terms)

    # =========================
    # 1. Đăng ký doanh nghiệp
    # =========================
    if has_all("hồ sơ đăng ký doanh nghiệp") and has_any(
        "chưa hợp lệ", "không hợp lệ", "sửa hồ sơ", "bổ sung hồ sơ"
    ):
        expansions.extend([
            "168/2025/NĐ-CP Điều 32",
            "thông báo yêu cầu sửa đổi bổ sung hồ sơ đăng ký doanh nghiệp",
            "hồ sơ chưa hợp lệ cơ quan đăng ký kinh doanh thông báo bằng văn bản",
        ])

    if "tên doanh nghiệp bằng tiếng nước ngoài" in q:
        expansions.extend([
            "59/2020/QH14 Điều 39",
            "tên doanh nghiệp bằng tiếng nước ngoài tên viết tắt doanh nghiệp",
        ])

    if has_any("mã số thuế của doanh nghiệp", "mã số doanh nghiệp"):
        expansions.extend([
            "168/2025/NĐ-CP Điều 8",
            "mã số doanh nghiệp đồng thời là mã số thuế của doanh nghiệp",
        ])

    # =========================
    # 2. Hỗ trợ DNNVV
    # =========================
    if has_any("doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "dnnvv"):
        expansions.extend([
            "Luật Hỗ trợ doanh nghiệp nhỏ và vừa 04/2017/QH14",
            "Nghị định 80/2021/NĐ-CP hỗ trợ doanh nghiệp nhỏ và vừa",
        ])

    if has_any("cơ sở ươm tạo", "ươm tạo", "khu làm việc chung", "cơ sở kỹ thuật"):
        expansions.extend([
            "04/2017/QH14 Điều 12",
            "hỗ trợ cơ sở ươm tạo cơ sở kỹ thuật khu làm việc chung",
            "hỗ trợ thuế đất đai cơ sở ươm tạo khu làm việc chung",
        ])

    if has_all("số lao động tham gia bảo hiểm xã hội bình quân năm") and has_any(
        "doanh nghiệp nhỏ và vừa", "nhỏ và vừa"
    ):
        expansions.extend([
            "04/2017/QH14 Điều 4",
            "tiêu chí doanh nghiệp nhỏ và vừa số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người",
        ])

    if has_any("hỗ trợ tư vấn", "mức hỗ trợ tư vấn", "mạng lưới tư vấn viên"):
        expansions.extend([
            "80/2021/NĐ-CP Điều 13",
            "hỗ trợ tư vấn doanh nghiệp nhỏ và vừa mạng lưới tư vấn viên mức hỗ trợ",
            "doanh nghiệp siêu nhỏ 50 triệu doanh nghiệp nhỏ 100 triệu doanh nghiệp vừa 150 triệu",
        ])

    if has_all("chi phí", "tư vấn viên") and has_any("hỗ trợ tư vấn", "ngân sách nhà nước"):
        expansions.extend([
            "52/2023/TT-BTC Điều 7",
            "chi phí tư vấn viên hỗ trợ tư vấn doanh nghiệp nhỏ và vừa",
            "chi phí theo hợp đồng tư vấn không phải chi phí học viên",
        ])

    if has_any("quỹ bảo lãnh tín dụng") and has_any("điều kiện", "cấp bảo lãnh"):
        expansions.extend([
            "34/2018/NĐ-CP Điều 16",
            "điều kiện cấp bảo lãnh tín dụng doanh nghiệp nhỏ và vừa",
            "phương án sản xuất kinh doanh khả thi có khả năng hoàn trả vốn vay",
        ])

    if has_all("bộ tài chính") and has_any("doanh nghiệp siêu nhỏ", "thuế", "kế toán"):
        expansions.extend([
            "04/2017/QH14 Điều 23",
            "Bộ Tài chính hướng dẫn thủ tục hành chính thuế chế độ kế toán doanh nghiệp siêu nhỏ",
            "132/2018/TT-BTC chế độ kế toán doanh nghiệp siêu nhỏ",
        ])

    # =========================
    # 3. Thuế, hóa đơn, chứng từ
    # =========================
    if "phạm vi đăng ký thuế" in q:
        expansions.extend([
            "38/2019/QH14 Điều 30",
            "105/2020/TT-BTC phạm vi đăng ký thuế cấp mã số thuế thay đổi thông tin đăng ký thuế chấm dứt hiệu lực mã số thuế",
        ])

    if has_any("đăng ký sử dụng hóa đơn điện tử"):
        expansions.extend([
            "123/2020/NĐ-CP Điều 15",
            "đăng ký thay đổi nội dung đăng ký sử dụng hóa đơn điện tử",
        ])

    if has_any("ngừng sử dụng hóa đơn điện tử", "buộc phải ngừng sử dụng hóa đơn điện tử"):
        expansions.extend([
            "123/2020/NĐ-CP Điều 16",
            "các trường hợp ngừng sử dụng hóa đơn điện tử",
        ])

    if has_any("loại hóa đơn điện tử", "những loại hóa đơn điện tử"):
        expansions.extend([
            "38/2019/QH14 Điều 89",
            "hóa đơn điện tử có mã của cơ quan thuế hóa đơn điện tử không có mã của cơ quan thuế",
        ])

    if has_all("hóa đơn điện tử") and has_any("sai tên", "sai địa chỉ"):
        expansions.extend([
            "123/2020/NĐ-CP Điều 19",
            "xử lý hóa đơn điện tử có sai sót sai tên địa chỉ người mua không sai mã số thuế",
        ])

    if has_any("hóa đơn điện tử không có mã", "không có mã của cơ quan thuế"):
        expansions.extend([
            "38/2019/QH14 Điều 91",
            "123/2020/NĐ-CP Điều 18",
            "sử dụng hóa đơn điện tử không có mã của cơ quan thuế điều kiện hạ tầng công nghệ thông tin phần mềm kế toán truyền dữ liệu",
        ])

    if has_all("biện pháp cưỡng chế") and has_any("nợ thuế", "cơ quan thuế"):
        expansions.extend([
            "38/2019/QH14 Điều 125",
            "biện pháp cưỡng chế thi hành quyết định hành chính về quản lý thuế",
            "trích tiền từ tài khoản phong tỏa tài khoản khấu trừ lương ngừng sử dụng hóa đơn kê biên tài sản thu hồi giấy chứng nhận",
        ])

    # =========================
    # 4. Lao động, BHXH, xử phạt
    # =========================
    if has_any("giữ bản chính") and has_any("bằng cấp", "văn bằng", "chứng chỉ", "giấy tờ tùy thân"):
        expansions.extend([
            "12/2022/NĐ-CP Điều 9",
            "giữ bản chính giấy tờ tùy thân văn bằng chứng chỉ của người lao động",
            "buộc trả lại bản chính giấy tờ tùy thân văn bằng chứng chỉ",
        ])

    if has_all("chậm đóng") and has_any("bảo hiểm xã hội", "bhxh"):
        expansions.extend([
            "12/2022/NĐ-CP Điều 39",
            "chậm đóng bảo hiểm xã hội bắt buộc bảo hiểm thất nghiệp buộc đóng đủ nộp lãi",
        ])

    if has_any("không trả sổ bảo hiểm xã hội", "không trả sổ bhxh") or has_all(
        "sổ bảo hiểm xã hội", "chấm dứt hợp đồng"
    ):
        expansions.extend([
            "12/2022/NĐ-CP Điều 12",
            "chấm dứt hợp đồng lao động xác nhận thời gian đóng bảo hiểm xã hội trả lại sổ bảo hiểm xã hội giấy tờ",
        ])

    if has_all("hình thức xử phạt chính") and has_any("lao động", "bảo hiểm xã hội"):
        expansions.extend([
            "12/2022/NĐ-CP Điều 3",
            "hình thức xử phạt chính cảnh cáo phạt tiền",
        ])

    if has_any("khám sức khỏe định kỳ") and has_any("xử phạt", "bị phạt", "phạt"):
        expansions.extend([
            "12/2022/NĐ-CP Điều 22",
            "không tổ chức khám sức khỏe định kỳ cho người lao động xử phạt",
        ])

    if has_any("cán bộ công đoàn", "công đoàn cấp trên") and has_any(
        "tuyên truyền", "thành lập công đoàn", "hướng dẫn người lao động"
    ):
        expansions.extend([
            "50/2024/QH15 Điều 19",
            "12/2022/NĐ-CP Điều 35",
            "công đoàn cấp trên trực tiếp cơ sở tuyên truyền vận động hướng dẫn thành lập công đoàn",
            "cản trở người lao động thành lập gia nhập hoạt động công đoàn",
        ])

    # =========================
    # 5. Sở hữu trí tuệ
    # =========================
    if has_all("biện pháp dân sự") and has_any("sở hữu trí tuệ", "xâm phạm quyền"):
        expansions.extend([
            "50/2005/QH11 Điều 202",
            "biện pháp dân sự xâm phạm quyền sở hữu trí tuệ buộc chấm dứt hành vi xâm phạm xin lỗi cải chính bồi thường thiệt hại tiêu hủy hàng hóa",
        ])

    if has_any("hợp đồng chuyển nhượng quyền sở hữu công nghiệp") and has_any(
        "hiệu lực", "có hiệu lực"
    ):
        expansions.extend([
            "50/2005/QH11 Điều 148",
            "hiệu lực hợp đồng chuyển giao quyền sở hữu công nghiệp đăng ký tại cơ quan quản lý nhà nước",
        ])

    if not expansions:
        return query

    # Khử trùng lặp để query không bị kéo quá dài
    seen = set()
    deduped = []
    for item in expansions:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return query + " " + " ".join(deduped)

def detect_query_signals(query: str) -> dict:
    query_lower = query.lower()
    query_upper = query.upper()

    document_numbers = re.findall(
        r"\d+/\d{4}/[A-ZÀ-ỸĐ\-]+",
        query_upper,
        flags=re.IGNORECASE,
    )

    articles = re.findall(
        r"(?:điều|dieu)\s+(\d+[a-zA-Z]?)",
        query_lower,
        flags=re.IGNORECASE,
    )

    clauses = re.findall(
        r"(?:khoản|khoan)\s+(\d+[a-zA-Z]?)",
        query_lower,
        flags=re.IGNORECASE,
    )

    return {
        "document_numbers": [d.upper() for d in document_numbers],
        "articles": [a.lower() for a in articles],
        "clauses": [c.lower() for c in clauses],
    }


def get_metadata_value(metadata: dict, keys: List[str], default=None):
    for key in keys:
        if key in metadata and metadata[key] not in [None, "", []]:
            return metadata[key]
    return default


def rule_boost(query: str, chunk: dict, query_signals: dict) -> float:
    boost = 0.0

    metadata = chunk.get("metadata", {})
    query_lower = query.lower()
    query_upper = query.upper()

    document_number = str(
        get_metadata_value(
            metadata,
            ["document_number", "doc_number", "so_hieu_van_ban"],
            "",
        )
    ).upper()

    article_number = str(
        get_metadata_value(
            metadata,
            ["article_number", "article_id", "dieu"],
            "",
        )
    ).lower()

    legal_reference_keys = metadata.get("legal_reference_keys", []) or []
    citation_aliases = metadata.get("citation_aliases", []) or []
    plain_language_aliases = metadata.get("plain_language_aliases", []) or []

    citation = chunk.get("citation", "") or ""
    retrieval_title = chunk.get("retrieval_title", "") or ""

    combined_refs = " ".join(
        [
            document_number,
            citation,
            retrieval_title,
            " ".join(legal_reference_keys),
            " ".join(citation_aliases),
            " ".join(plain_language_aliases),
        ]
    ).upper()

    # Boost nếu query nêu rõ số hiệu văn bản
    for doc_num in query_signals["document_numbers"]:
        if doc_num and doc_num in combined_refs:
            boost += 0.30

    # Boost nếu query nêu rõ Điều
    for article in query_signals["articles"]:
        if article and article == article_number:
            boost += 0.20

        if f"ĐIỀU {article.upper()}" in combined_refs:
            boost += 0.10

    # Boost nhẹ nếu title/citation chứa nhiều từ khóa domain rõ
    important_terms = [
        "đăng ký doanh nghiệp",
        "hồ sơ đăng ký doanh nghiệp",
        "hóa đơn điện tử",
        "bảo hiểm xã hội",
        "hợp đồng lao động",
        "dữ liệu cá nhân",
        "an ninh mạng",
        "đấu thầu",
        "đất đai",
        "môi trường",
        "an toàn thực phẩm",
    ]

    title_lower = f"{citation} {retrieval_title}".lower()

    domain_matched = False

    for term in important_terms:
        if term in query_lower and term in title_lower:
            domain_matched = True
            break

    if domain_matched:
        boost += 0.05

    return boost


def bm25_retrieve(query: str, bm25, top_k: int) -> List[Tuple[int, float]]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [(int(idx), float(score)) for idx, score in ranked]


def dense_retrieve(query: str, model, index, top_k: int) -> List[Tuple[int, float]]:
    query_embedding = model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(query_embedding, top_k)

    return [
        (int(idx), float(score))
        for idx, score in zip(ids[0], scores[0])
    ]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    chunks: list,
    bm25,
    dense_model,
    dense_index,
    bm25_top_k: int = 100,
    dense_top_k: int = 100,
    final_top_k: int = 15,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    group_by_parent: bool = True,
):
    expanded_query = expand_query(query)

    bm25_results = bm25_retrieve(expanded_query, bm25, bm25_top_k)
    dense_results = dense_retrieve(expanded_query, dense_model, dense_index, dense_top_k)

    scores: Dict[int, dict] = {}

    for rank, (idx, raw_score) in enumerate(bm25_results, start=1):
        if idx not in scores:
            scores[idx] = {
                "idx": idx,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_raw": None,
                "dense_raw": None,
                "rrf_score": 0.0,
                "rule_boost": 0.0,
                "final_score": 0.0,
            }

        scores[idx]["bm25_rank"] = rank
        scores[idx]["bm25_raw"] = raw_score
        scores[idx]["rrf_score"] += bm25_weight * rrf_score(rank, rrf_k)

    for rank, (idx, raw_score) in enumerate(dense_results, start=1):
        if idx not in scores:
            scores[idx] = {
                "idx": idx,
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_raw": None,
                "dense_raw": None,
                "rrf_score": 0.0,
                "rule_boost": 0.0,
                "final_score": 0.0,
            }

        scores[idx]["dense_rank"] = rank
        scores[idx]["dense_raw"] = raw_score
        scores[idx]["rrf_score"] += dense_weight * rrf_score(rank, rrf_k)

    query_signals = detect_query_signals(expanded_query)

    for idx, item in scores.items():
        chunk = chunks[idx]
        boost = rule_boost(expanded_query, chunk, query_signals)
        item["rule_boost"] = boost
        item["final_score"] = item["rrf_score"] + boost

    ranked_items = sorted(
        scores.values(),
        key=lambda x: x["final_score"],
        reverse=True,
    )

    if group_by_parent:
        grouped = {}

        for item in ranked_items:
            chunk = chunks[item["idx"]]
            parent_id = chunk.get("parent_article_id") or chunk.get("chunk_id")

            if parent_id not in grouped:
                grouped[parent_id] = item

        ranked_items = list(grouped.values())

    ranked_items = ranked_items[:final_top_k]

    output = []

    for rank, item in enumerate(ranked_items, start=1):
        chunk = chunks[item["idx"]]
        metadata = chunk.get("metadata", {})

        output.append({
            "rank": rank,
            "final_score": float(item["final_score"]),
            "rrf_score": float(item["rrf_score"]),
            "rule_boost": float(item["rule_boost"]),
            "bm25_rank": item["bm25_rank"],
            "dense_rank": item["dense_rank"],
            "bm25_raw": item["bm25_raw"],
            "dense_raw": item["dense_raw"],
            "chunk_id": chunk.get("chunk_id"),
            "parent_article_id": chunk.get("parent_article_id"),
            "chunk_level": chunk.get("chunk_level"),
            "citation": chunk.get("citation"),
            "retrieval_title": chunk.get("retrieval_title"),
            "document_number": metadata.get("document_number"),
            "document_title": metadata.get("document_title"),
            "document_type": metadata.get("document_type"),
            "legal_reference_keys": metadata.get("legal_reference_keys", []),
            "content_preview": chunk.get("content", "")[:700],
        })

    return output


def print_results(query: str, results: list):
    print("\n" + "=" * 120)
    print("[QUERY]", query)
    print("=" * 120)

    for r in results:
        print("-" * 120)
        print("Rank       :", r["rank"])
        print("Final score:", r["final_score"])
        print("RRF score  :", r["rrf_score"])
        print("Rule boost :", r["rule_boost"])
        print("BM25 rank  :", r["bm25_rank"], "| raw:", r["bm25_raw"])
        print("Dense rank :", r["dense_rank"], "| raw:", r["dense_raw"])
        print("Citation   :", r["citation"])
        print("Title      :", r["retrieval_title"])
        print("Ref keys   :", r["legal_reference_keys"])
        print("Preview    :", r["content_preview"][:350])

    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/kaggle/working/artifacts",
        help="Thư mục chứa bm25.pkl, dense_faiss.index, chunks.pkl",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Tên dense embedding model",
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
        "--final_top_k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--test_query",
        type=str,
        action="append",
        default=None,
        help="Có thể truyền nhiều --test_query",
    )

    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    if not bm25_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {bm25_path}. Hãy chạy 02_build_bm25.py trước.")

    if not dense_index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {dense_index_path}. Hãy chạy 03_build_dense_index.py trước.")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {chunks_path}. Hãy chạy 02 hoặc 03 trước.")

    print("[INFO] Artifact dir:", artifact_dir)
    print("[INFO] Loading BM25:", bm25_path)
    bm25 = load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = load_pickle(chunks_path)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] Using device:", device)
    print("[INFO] Loading dense model:", args.model_name)

    dense_model = SentenceTransformer(args.model_name, device=device)

    queries = args.test_query or [
        "Hồ sơ đăng ký doanh nghiệp bằng tiếng nước ngoài có cần dịch công chứng không?",
        "Tài liệu tiếng Anh trong hồ sơ thành lập công ty có phải dịch sang tiếng Việt không?",
        "Người lao động bị sa thải trong trường hợp nào?",
        "Doanh nghiệp sử dụng hóa đơn điện tử khi nào?",
        "Quy định về xử lý dữ liệu cá nhân của doanh nghiệp là gì?",
    ]

    all_results = []

    for query in queries:
        results = hybrid_search(
            query=query,
            chunks=chunks,
            bm25=bm25,
            dense_model=dense_model,
            dense_index=dense_index,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            final_top_k=args.final_top_k,
        )

        print_results(query, results)

        all_results.append({
            "query": query,
            "results": results,
        })

    save_json(artifact_dir / "hybrid_test_results.json", all_results)

    print("[DONE] Saved hybrid test results to:", artifact_dir / "hybrid_test_results.json")


if __name__ == "__main__":
    main()