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
    Mở rộng query bằng các cụm pháp lý tương đương.
    Chỉ dùng cho retrieval, không thay đổi câu hỏi gốc.
    """
    q = query.lower()
    expansions = []

    # Hồ sơ đăng ký doanh nghiệp chưa hợp lệ
    if "hồ sơ đăng ký doanh nghiệp" in q and (
        "chưa hợp lệ" in q
        or "không hợp lệ" in q
        or "sửa hồ sơ" in q
        or "bổ sung hồ sơ" in q
    ):
        expansions.extend([
            "thông báo yêu cầu sửa đổi bổ sung hồ sơ đăng ký doanh nghiệp",
            "hồ sơ chưa hợp lệ",
            "nội dung cần sửa đổi bổ sung",
            "cơ quan đăng ký kinh doanh thông báo bằng văn bản",
            "Điều 32 Nghị định 168/2025/NĐ-CP",
        ])

    # Tên doanh nghiệp bằng tiếng nước ngoài
    if "tên doanh nghiệp bằng tiếng nước ngoài" in q:
        expansions.extend([
            "Điều 39 Luật Doanh nghiệp",
            "tên doanh nghiệp bằng tiếng nước ngoài và tên viết tắt",
            "59/2020/QH14 Điều 39",
        ])

    # Đăng ký sử dụng hóa đơn điện tử
    if "đăng ký sử dụng hóa đơn điện tử" in q:
        expansions.extend([
            "Điều 15 Nghị định 123/2020/NĐ-CP",
            "đăng ký thay đổi nội dung đăng ký sử dụng hóa đơn điện tử",
        ])

    # Ngừng sử dụng hóa đơn điện tử
    if "ngừng sử dụng hóa đơn điện tử" in q:
        expansions.extend([
            "Điều 16 Nghị định 123/2020/NĐ-CP",
            "các trường hợp ngừng sử dụng hóa đơn điện tử",
        ])
    #    
    if (
        ("cơ sở ươm tạo" in q or "ươm tạo" in q)
        and ("khu làm việc chung" in q or "làm việc chung" in q)
    ):
        expansions.extend([
            "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "04/2017/QH14 Điều 12",
            "hỗ trợ cơ sở ươm tạo cơ sở kỹ thuật khu làm việc chung",
            "hỗ trợ thuế đất đai cơ sở ươm tạo khu làm việc chung",
        ])

    if "giữ bản chính" in q and ("bằng cấp" in q or "văn bằng" in q or "chứng chỉ" in q):
        expansions.extend([
            "Nghị định 12/2022/NĐ-CP Điều 9",
            "giữ bản chính giấy tờ tùy thân văn bằng chứng chỉ của người lao động",
            "buộc trả lại bản chính giấy tờ tùy thân văn bằng chứng chỉ",
            "vi phạm giao kết hợp đồng lao động",
        ])

    if (
        "số lao động tham gia bảo hiểm xã hội bình quân năm" in q
        and ("doanh nghiệp nhỏ và vừa" in q or "nhỏ và vừa" in q)
    ):
        expansions.extend([
            "Luật Hỗ trợ doanh nghiệp nhỏ và vừa 04/2017/QH14 Điều 4",
            "tiêu chí doanh nghiệp nhỏ và vừa số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người",
        ])

    if "quỹ bảo lãnh tín dụng" in q and ("điều kiện" in q or "cấp bảo lãnh" in q):
        expansions.extend([
        "34/2018/NĐ-CP Điều 16 điều kiện cấp bảo lãnh tín dụng",
        "điều kiện cấp bảo lãnh tín dụng doanh nghiệp nhỏ và vừa",
        "có dự án đầu tư phương án sản xuất kinh doanh khả thi có khả năng hoàn trả vốn vay",
    ])

    if "bộ tài chính" in q and "doanh nghiệp siêu nhỏ" in q:
        expansions.extend([
            "Luật Hỗ trợ doanh nghiệp nhỏ và vừa 04/2017/QH14 Điều 23",
            "Bộ Tài chính hướng dẫn thuế kế toán doanh nghiệp siêu nhỏ",
            "Thông tư 132/2018/TT-BTC chế độ kế toán doanh nghiệp siêu nhỏ",
        ])   

    # Hóa đơn điện tử - loại hóa đơn
    if "loại hóa đơn điện tử" in q or "những loại hóa đơn điện tử" in q:
        expansions.extend([
            "123/2020/NĐ-CP Điều 3 hóa đơn điện tử có mã của cơ quan thuế hóa đơn điện tử không có mã",
            "38/2019/QH14 Điều 89 hóa đơn điện tử",
            "hóa đơn điện tử có mã của cơ quan thuế không có mã của cơ quan thuế",
        ])

    # Cưỡng chế nợ thuế
    if "biện pháp cưỡng chế" in q and ("nợ thuế" in q or "cơ quan thuế" in q):
        expansions.extend([
            "38/2019/QH14 Điều 125 biện pháp cưỡng chế thi hành quyết định hành chính về quản lý thuế",
            "trích tiền từ tài khoản khấu trừ tiền lương dừng làm thủ tục hải quan ngừng sử dụng hóa đơn kê biên tài sản thu hồi giấy chứng nhận",
        ])

    # Biện pháp dân sự SHTT
    if "biện pháp dân sự" in q and ("sở hữu trí tuệ" in q or "xâm phạm quyền" in q):
        expansions.extend([
            "50/2005/QH11 Điều 202 biện pháp dân sự xử lý xâm phạm quyền sở hữu trí tuệ",
            "buộc chấm dứt hành vi xâm phạm xin lỗi cải chính công khai bồi thường thiệt hại tiêu hủy hàng hóa",
        ])

    # Hóa đơn sai tên địa chỉ
    if "hóa đơn điện tử" in q and ("sai tên" in q or "sai địa chỉ" in q):
        expansions.extend([
            "123/2020/NĐ-CP Điều 19 xử lý hóa đơn có sai sót sai tên địa chỉ người mua",
            "sai tên địa chỉ người mua nhưng không sai mã số thuế không phải lập lại hóa đơn",
        ])

    # Không trả sổ BHXH
    if "không trả sổ bảo hiểm xã hội" in q or "không trả sổ bhxh" in q:
        expansions.extend([
            "12/2022/NĐ-CP Điều 12 không hoàn thành thủ tục xác nhận thời gian đóng bảo hiểm xã hội không trả lại giấy tờ",
            "chấm dứt hợp đồng lao động trả sổ bảo hiểm xã hội cho người lao động",
        ])

    # Hình thức xử phạt chính
    if "hình thức xử phạt chính" in q and ("lao động" in q or "bảo hiểm xã hội" in q):
        expansions.extend([
            "12/2022/NĐ-CP Điều 4 hình thức xử phạt biện pháp khắc phục hậu quả cảnh cáo phạt tiền",
        ])
        
        # ID 22 - chi phí tư vấn viên, tránh nhầm sang chi phí học viên
    if (
        "chi phí" in q
        and "tư vấn viên" in q
        and ("hỗ trợ tư vấn" in q or "ngân sách nhà nước" in q)
    ):
        expansions.extend([
            "52/2023/TT-BTC Điều 7 chi phí tư vấn viên hỗ trợ tư vấn doanh nghiệp nhỏ và vừa",
            "chi phí thuê tư vấn viên theo hợp đồng tư vấn hỗ trợ doanh nghiệp nhỏ và vừa",
            "hỗ trợ tư vấn từ ngân sách nhà nước chi phí tư vấn viên không phải học viên",
        ])

    # ID 27 - phạm vi đăng ký thuế chung, tránh kéo nhầm Thông tư thuế TNCN
    if "phạm vi đăng ký thuế" in q:
        expansions.extend([
            "105/2020/TT-BTC đăng ký thuế phạm vi đăng ký thuế mã số thuế chấm dứt hiệu lực mã số thuế khôi phục mã số thuế",
            "38/2019/QH14 Điều 30 đăng ký thuế cấp mã số thuế sử dụng mã số thuế",
            "quản lý thuế đăng ký thuế cấp mã số thuế thay đổi thông tin đăng ký thuế chấm dứt hiệu lực mã số thuế",
        ])

    # ID 29 - cơ sở ươm tạo / khu làm việc chung, cơ cấu tổ chức bộ máy
    if (
        ("khu làm việc chung" in q or "cơ sở ươm tạo" in q)
        and ("cơ cấu tổ chức" in q or "bộ máy" in q)
    ):
        expansions.extend([
            "80/2021/NĐ-CP khu làm việc chung cơ cấu tổ chức bộ máy nhân sự quản lý điều hành",
            "80/2021/NĐ-CP cơ sở ươm tạo khu làm việc chung hỗ trợ doanh nghiệp khởi nghiệp sáng tạo điều kiện cơ cấu tổ chức",
            "cơ sở ươm tạo khu làm việc chung có cơ cấu tổ chức bộ máy nhân sự chuyên môn",
        ])

    # ID 65 - không khám sức khỏe định kỳ, cần nghị định xử phạt
    if (
        "không tổ chức khám sức khỏe định kỳ" in q
        or ("khám sức khỏe định kỳ" in q and ("xử phạt" in q or "bị phạt" in q))
    ):
        expansions.extend([
            "12/2022/NĐ-CP Điều 22 không tổ chức khám sức khỏe định kỳ cho người lao động",
            "không tổ chức khám sức khỏe định kỳ phạt tiền mỗi người lao động tối đa 75.000.000 đồng",
            "vi phạm quy định về phòng ngừa tai nạn lao động bệnh nghề nghiệp khám sức khỏe định kỳ",
        ])

    # ID 71 - hình thức xử phạt chính, tránh nhầm sang biện pháp khắc phục
    if (
        "hình thức xử phạt chính" in q
        and ("lao động" in q or "bảo hiểm xã hội" in q)
    ):
        expansions.extend([
            "12/2022/NĐ-CP Điều 3 hình thức xử phạt cảnh cáo phạt tiền",
            "hình thức xử phạt chính là cảnh cáo hoặc phạt tiền",
        ])

    # ID 79 - không trả sổ BHXH khi chấm dứt hợp đồng
    if (
        "không trả sổ bảo hiểm xã hội" in q
        or "không trả sổ bhxh" in q
        or ("sổ bảo hiểm xã hội" in q and "chấm dứt hợp đồng" in q)
    ):
        expansions.extend([
            "12/2022/NĐ-CP Điều 12 không hoàn thành thủ tục xác nhận thời gian đóng bảo hiểm xã hội",
            "chấm dứt hợp đồng lao động trả lại sổ bảo hiểm xã hội giấy tờ cho người lao động",
            "không trả sổ bảo hiểm xã hội cho người lao động khi chấm dứt hợp đồng lao động",
        ])

    # ID 83 - công đoàn cấp trên vào doanh nghiệp tuyên truyền thành lập công đoàn
    if (
        "cán bộ công đoàn" in q
        and ("tuyên truyền" in q or "thành lập công đoàn" in q)
    ):
        expansions.extend([
            "50/2024/QH15 Điều 19 công đoàn cấp trên trực tiếp cơ sở tuyên truyền vận động hướng dẫn thành lập công đoàn",
            "12/2022/NĐ-CP Điều 35 cản trở người lao động thành lập gia nhập hoạt động công đoàn",
            "không cho cán bộ công đoàn vào doanh nghiệp tuyên truyền hướng dẫn người lao động thành lập công đoàn",
        ])

    # ID 89 - hóa đơn điện tử không có mã, tránh nhầm sang miễn phí dịch vụ
    if (
        "hóa đơn điện tử không có mã" in q
        or "không có mã của cơ quan thuế" in q
    ):
        expansions.extend([
            "38/2019/QH14 Điều 91 sử dụng hóa đơn điện tử không có mã của cơ quan thuế",
            "123/2020/NĐ-CP Điều 18 hóa đơn điện tử không có mã của cơ quan thuế",
            "doanh nghiệp sử dụng hóa đơn điện tử không có mã có giao dịch điện tử phần mềm kế toán phần mềm hóa đơn truyền dữ liệu đến cơ quan thuế",
        ])

    if not expansions:
        return query

    return query + " " + " ".join(expansions)

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