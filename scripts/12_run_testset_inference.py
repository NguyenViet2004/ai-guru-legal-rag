import argparse
import gc
import importlib.util
import json
import re
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


# ============================================================
# Basic IO / dynamic imports
# ============================================================

def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_outputs(output_file: Path):
    if not output_file.exists():
        return {}

    try:
        data = load_json(output_file)
    except Exception:
        return {}

    if isinstance(data, list):
        return {int(item["id"]): item for item in data if "id" in item}

    return {}


# ============================================================
# Legal reference formatting
# ============================================================

def safere(text: str) -> str:
    return re.escape(str(text))


def normalize_doc_type(doc_type: str) -> str:
    if not doc_type:
        return "Văn bản"

    t = " ".join(str(doc_type).split()).strip().lower()

    mapping = {
        "luật": "Luật",
        "bộ luật": "Bộ luật",
        "nghị định": "Nghị định",
        "thông tư": "Thông tư",
        "pháp lệnh": "Pháp lệnh",
        "nghị quyết": "Nghị quyết",
        "quyết định": "Quyết định",
    }

    return mapping.get(t, str(doc_type).strip().title())


def clean_doc_title(title: str, doc_no: str = "") -> str:
    if not title:
        return ""

    title = " ".join(str(title).split()).strip()

    if doc_no:
        title = re.sub(
            rf"\s+số\s+{safere(doc_no)}\b",
            " ",
            title,
            flags=re.IGNORECASE,
        )
        title = re.sub(
            rf"\b{safere(doc_no)}\b",
            " ",
            title,
            flags=re.IGNORECASE,
        )

    title = re.sub(r"\s+", " ", title).strip(" -|")
    return title


def parse_doc_from_retrieval_title(retrieval_title: str, doc_no: str):
    if not retrieval_title:
        return {
            "document_type": "Văn bản",
            "document_title": "",
        }

    head = retrieval_title.split("|")[0].strip()

    doc_type_candidates = [
        "Bộ Luật",
        "Luật",
        "Nghị Định",
        "Thông Tư",
        "Pháp Lệnh",
        "Nghị Quyết",
        "Quyết Định",
    ]

    for cand in doc_type_candidates:
        if head.lower().startswith(cand.lower()):
            rest = head[len(cand):].strip()

            if doc_no:
                rest = re.sub(
                    rf"\s+số\s+{safere(doc_no)}",
                    "",
                    rest,
                    flags=re.IGNORECASE,
                )
                rest = rest.replace(doc_no, "")

            return {
                "document_type": normalize_doc_type(cand),
                "document_title": clean_doc_title(rest, doc_no),
            }

    return {
        "document_type": "Văn bản",
        "document_title": clean_doc_title(head, doc_no),
    }


def normalize_article_ref(ref: str):
    if not ref:
        return None

    parts = str(ref).split("|")
    if len(parts) < 2:
        return None

    doc_no = parts[0].strip()
    article = parts[1].strip()

    if not doc_no or not article.lower().startswith("điều"):
        return None

    normalized_ref = f"{doc_no}|{article}"

    return {
        "doc_no": doc_no,
        "article": article,
        "normalized_ref": normalized_ref,
    }


def get_doc_info_for_ref(ref: str, context: dict, lookup: dict):
    norm = normalize_article_ref(ref)
    if not norm:
        return None

    doc_no = norm["doc_no"]
    chunk = None

    if ref in lookup["by_ref"]:
        chunk = lookup["by_ref"][ref]

    if chunk is None and norm["normalized_ref"] in lookup["by_ref"]:
        chunk = lookup["by_ref"][norm["normalized_ref"]]

    metadata = chunk.get("metadata", {}) if chunk else {}

    doc_type = metadata.get("document_type") or metadata.get("type") or ""
    doc_title = metadata.get("document_title") or metadata.get("title") or ""

    if not doc_type or not doc_title:
        parsed = parse_doc_from_retrieval_title(
            context.get("retrieval_title", ""),
            doc_no,
        )
        doc_type = doc_type or parsed["document_type"]
        doc_title = doc_title or parsed["document_title"]

    doc_type = normalize_doc_type(doc_type)
    doc_title = clean_doc_title(doc_title, doc_no)

    if doc_title:
        formatted_doc_name = f"{doc_type} {doc_no} {doc_title}".strip()
    else:
        formatted_doc_name = f"{doc_type} {doc_no}".strip()

    return {
        "doc_no": doc_no,
        "document_name": formatted_doc_name,
        "article": norm["article"],
        "normalized_ref": norm["normalized_ref"],
    }


def make_relevant_fields(contexts: list[dict], lookup: dict, max_docs: int, max_articles: int):
    relevant_docs = []
    relevant_articles = []

    seen_docs = set()
    seen_articles = set()

    for ctx in contexts:
        refs = ctx.get("legal_reference_keys", []) or []

        for ref in refs:
            info = get_doc_info_for_ref(ref, ctx, lookup)
            if not info:
                continue

            doc_item = f"{info['doc_no']}|{info['document_name']}"
            article_item = f"{info['doc_no']}|{info['document_name']}|{info['article']}"

            if doc_item not in seen_docs and len(relevant_docs) < max_docs:
                seen_docs.add(doc_item)
                relevant_docs.append(doc_item)

            if article_item not in seen_articles and len(relevant_articles) < max_articles:
                seen_articles.add(article_item)
                relevant_articles.append(article_item)

    return relevant_docs, relevant_articles


# ============================================================
# AI signal / domain logic
# ============================================================

def is_explicit_ai_question(text: str) -> bool:
    """
    Chỉ nhận AI khi là ngữ cảnh công nghệ thật sự.
    Không match chữ "ai" tiếng Việt trong câu hỏi kiểu "ai phải nộp".
    """
    raw = str(text or "")
    q = raw.lower()

    explicit_phrases = [
        "hệ thống ai",
        "mô hình ai",
        "ứng dụng ai",
        "công cụ ai",
        "sản phẩm ai",
        "trí tuệ nhân tạo",
        "artificial intelligence",
        "công nghiệp công nghệ số",
        "luật công nghiệp công nghệ số",
    ]

    if any(x in q for x in explicit_phrases):
        return True

    if re.search(r"(^|[^A-Za-z])AI([^A-Za-z]|$)", raw):
        tech_context = [
            "hệ thống",
            "mô hình",
            "ứng dụng",
            "công nghệ",
            "dữ liệu",
            "thuật toán",
            "huấn luyện",
            "rủi ro",
            "phân loại",
            "triển khai",
        ]
        return any(x in q for x in tech_context)

    return False


def infer_question_domains(question: str) -> set[str]:
    q = question.lower()
    domains = set()

    if any(x in q for x in [
        "hộ kinh doanh",
        "đăng ký doanh nghiệp",
        "mã số doanh nghiệp",
        "chi nhánh",
        "văn phòng đại diện",
        "vốn điều lệ",
        "người đại diện theo pháp luật",
    ]):
        domains.add("business_registration")

    if any(x in q for x in [
        "doanh nghiệp nhỏ và vừa",
        "nhỏ và vừa",
        "dnnvv",
        "quỹ bảo lãnh tín dụng",
        "quỹ phát triển doanh nghiệp",
        "khởi nghiệp sáng tạo",
        "chuỗi giá trị",
        "cụm liên kết ngành",
    ]):
        domains.add("sme_support")

    if any(x in q for x in [
        "thuế",
        "hóa đơn",
        "hoá đơn",
        "chứng từ",
        "biên lai",
        "mã số thuế",
        "cơ quan thuế",
        "lệ phí trước bạ",
        "tiêu thụ đặc biệt",
        "khai thiếu",
        "khai sai",
        "trốn thuế",
        "ấn định thuế",
        "gia hạn nộp thuế",
    ]):
        domains.add("tax_invoice")

    if any(x in q for x in [
        "lao động",
        "bảo hiểm xã hội",
        "bhxh",
        "bảo hiểm thất nghiệp",
        "công đoàn",
        "hợp đồng lao động",
        "tiền lương",
        "đình công",
        "an toàn vệ sinh lao động",
        "tai nạn lao động",
        "bệnh nghề nghiệp",
        "khám sức khỏe",
        "kỷ luật lao động",
        "thai sản",
    ]):
        domains.add("labor")

    if any(x in q for x in [
        "sở hữu trí tuệ",
        "sở hữu công nghiệp",
        "nhãn hiệu",
        "sáng chế",
        "kiểu dáng công nghiệp",
        "chỉ dẫn địa lý",
        "quyền tác giả",
        "quyền liên quan",
        "văn bằng bảo hộ",
        "tên thương mại",
        "giống cây trồng",
    ]):
        domains.add("ip")

    commercial_context = any(x in q for x in [
        "luật thương mại",
        "hợp đồng thương mại",
        "mua bán hàng hóa",
        "mua bán hàng hoá",
        "bên bán",
        "bên mua",
        "giao hàng",
        "giao hàng hóa",
        "giao hàng hoá",
        "thanh toán hàng hóa",
        "thanh toán hàng hoá",
        "đại lý thương mại",
        "môi giới thương mại",
        "nhượng quyền thương mại",
        "khuyến mại",
        "hội chợ",
        "triển lãm thương mại",
        "gia công hàng hóa",
        "gia công hàng hoá",
        "sở giao dịch hàng hóa",
        "sở giao dịch hàng hoá",
        "thương nhân",
    ])
    if commercial_context:
        domains.add("commercial")

    if any(x in q for x in [
        "dân sự",
        "ủy quyền",
        "uỷ quyền",
        "thế chấp",
        "tài sản bảo đảm",
        "bảo đảm thực hiện nghĩa vụ",
        "vận chuyển tài sản",
        "vận chuyển hàng hóa",
        "vận chuyển hàng hoá",
        "hợp đồng mua bán tài sản",
        "hoàn cảnh thay đổi cơ bản",
    ]):
        domains.add("civil")

    if any(x in q for x in [
        "người tiêu dùng",
        "khách hàng",
        "bán hàng từ xa",
        "dữ liệu khách hàng",
        "thông tin khách hàng",
        "hợp đồng theo mẫu",
        "điều kiện giao dịch chung",
        "sản phẩm có khuyết tật",
        "khiếu nại từ khách hàng",
        "người có ảnh hưởng",
        "kol",
    ]):
        domains.add("consumer")

    if any(x in q for x in [
        "kế toán",
        "kiểm toán",
        "báo cáo tài chính",
        "chứng chỉ kế toán viên",
        "chứng chỉ kiểm toán viên",
        "tài liệu kế toán",
        "chứng từ kế toán",
        "ghi sổ kế toán",
        "tẩy xóa",
        "sửa chữa chứng từ",
        "sổ kế toán",
        "khai man số liệu",
    ]):
        domains.add("accounting_audit")

    if any(x in q for x in [
        "môi trường",
        "giấy phép môi trường",
        "nước thải",
        "quan trắc",
        "nhãn sinh thái",
        "cụm công nghiệp",
    ]):
        domains.add("environment")

    if any(x in q for x in [
        "hải quan",
        "xuất khẩu",
        "nhập khẩu",
        "xuất xứ",
        "giấy chứng nhận xuất xứ",
        "quà biếu",
        "quà tặng",
    ]):
        domains.add("customs_trade")

    if any(x in q for x in [
        "an toàn thực phẩm",
        "thực phẩm chức năng",
        "trà sữa",
        "dụng cụ vệ sinh",
    ]):
        domains.add("food_safety")

    if any(x in q for x in [
        "trang thiết bị y tế",
        "thiết bị y tế",
    ]):
        domains.add("medical_device")

    if any(x in q for x in ["hiệp thương giá", "luật giá", "khung giá"]):
        domains.add("price")

    if any(x in q for x in ["du lịch", "lữ hành"]):
        domains.add("tourism")

    if is_explicit_ai_question(question) or any(x in q for x in ["an ninh mạng", "sự cố an ninh mạng"]):
        domains.add("digital_ai")

    if any(x in q for x in [
        "đấu thầu",
        "gói thầu",
        "hồ sơ dự thầu",
        "cptpp",
        "hiệp định cptpp",
    ]):
        domains.add("bidding")

    if any(x in q for x in ["xây dựng", "giấy phép xây dựng", "công trình"]):
        domains.add("construction")

    if any(x in q for x in [
        "vũ trường",
        "karaoke",
        "ngành nghề kinh doanh có điều kiện",
        "an ninh, trật tự",
    ]):
        domains.add("conditional_business")

    if any(x in q for x in [
        "trọng tài",
        "hội đồng trọng tài",
        "trung tâm trọng tài",
        "thỏa thuận trọng tài",
        "trọng tài viên",
        "biện pháp khẩn cấp tạm thời",
    ]):
        domains.add("arbitration")

    if any(x in q for x in [
        "quỹ đầu tư khởi nghiệp sáng tạo",
        "quỹ đầu tư khởi nghiệp",
        "công ty thực hiện quản lý quỹ",
        "nhà đầu tư góp vốn vào quỹ",
        "giải thể quỹ",
        "phân chia lợi tức",
    ]):
        domains.add("startup_fund")

    if any(x in q for x in [
        "công ty con",
        "góp vốn bằng tài sản",
        "biên bản giao nhận tài sản",
        "hội đồng quản trị",
        "đại hội đồng cổ đông",
        "thành viên hội đồng quản trị",
        "thành viên độc lập",
        "công ty tnhh",
        "công ty cổ phần",
    ]):
        domains.add("corporate_governance")

    return domains


DOMAIN_DOC_CODES = {
    "business_registration": ["59/2020/QH14", "168/2025/NĐ-CP"],
    "sme_support": [
        "04/2017/QH14", "80/2021/NĐ-CP", "06/2022/TT-BKHDT",
        "52/2023/TT-BTC", "34/2018/NĐ-CP", "39/2019/NĐ-CP",
        "45/2018/TT-NHNN", "57/2019/TT-BTC", "132/2018/TT-BTC",
    ],
    "tax_invoice": [
        "38/2019/QH14", "126/2020/NĐ-CP", "123/2020/NĐ-CP",
        "70/2025/NĐ-CP", "105/2020/TT-BTC", "80/2021/TT-BTC",
        "40/2021/TT-BTC", "320/2025/NĐ-CP", "67/2025/QH15",
        "125/2020/NĐ-CP", "181/2025/NĐ-CP",
    ],
    "labor": [
        "45/2019/QH14", "145/2020/NĐ-CP", "12/2022/NĐ-CP",
        "50/2024/QH15", "84/2015/QH13", "44/2016/NĐ-CP",
        "25/2022/TT-BLĐTBXH", "09/2020/TT-BLĐTBXH",
        "28/2021/TT-BLĐTBXH", "19/2016/TT-BYT",
    ],
    "ip": ["50/2005/QH11", "07/2022/QH15"],
    "commercial": ["36/2005/QH11", "35/2006/NĐ-CP", "09/2006/TT-BTM", "59/2015/TT-BCT", "69/2018/NĐ-CP"],
    "civil": ["91/2015/QH13", "21/2021/NĐ-CP"],
    "consumer": ["19/2023/QH15", "55/2024/NĐ-CP", "98/2020/NĐ-CP", "24/2025/NĐ-CP", "52/2013/NĐ-CP", "75/2025/QH15"],
    "accounting_audit": ["88/2015/QH13", "67/2011/QH12", "41/2018/NĐ-CP", "102/2021/NĐ-CP", "133/2016/TT-BTC", "132/2018/TT-BTC"],
    "environment": ["72/2020/QH14", "08/2022/NĐ-CP", "68/2017/NĐ-CP"],
    "customs_trade": ["54/2014/QH13", "107/2016/QH13", "69/2018/NĐ-CP"],
    "food_safety": ["115/2018/NĐ-CP"],
    "medical_device": ["98/2021/NĐ-CP", "07/2023/NĐ-CP"],
    "price": ["16/2023/QH15", "85/2024/NĐ-CP"],
    "tourism": ["09/2017/QH14", "168/2017/NĐ-CP"],
    "digital_ai": ["71/2025/QH15", "24/2018/QH14"],
    "bidding": ["22/2023/QH15", "214/2025/NĐ-CP"],
    "construction": ["50/2014/QH13", "06/2021/NĐ-CP"],
    "conditional_business": ["96/2016/NĐ-CP", "54/2019/NĐ-CP", "09/2017/QH14", "168/2017/NĐ-CP"],
    "arbitration": ["54/2010/QH12"],
    "startup_fund": ["38/2018/NĐ-CP", "04/2017/QH14", "80/2021/NĐ-CP"],
    "corporate_governance": ["59/2020/QH14", "168/2025/NĐ-CP"],
}


def result_text_for_domain(result: dict) -> str:
    refs = result.get("legal_reference_keys") or result.get("legal_refs") or []
    if isinstance(refs, str):
        refs = [refs]

    fields = [
        " ".join(refs),
        str(result.get("retrieval_title", "")),
        str(result.get("citation", "")),
        str(result.get("title", "")),
        str(result.get("doc_title", "")),
    ]

    meta = result.get("metadata") or {}
    fields.extend([
        str(meta.get("doc_no", "")),
        str(meta.get("doc_title", "")),
        str(meta.get("document_title", "")),
    ])

    return " ".join(fields)


def result_belongs_to_domains(result: dict, domains: set[str]) -> bool:
    if not domains:
        return True

    text = result_text_for_domain(result)

    for domain in domains:
        for code in DOMAIN_DOC_CODES.get(domain, []):
            if code in text:
                return True

    return False


def prioritize_results_by_domain(question: str, results: list[dict], min_domain_results: int = 3) -> list[dict]:
    domains = infer_question_domains(question)
    if not domains:
        return results

    good, bad = [], []
    for result in results:
        if result_belongs_to_domains(result, domains):
            good.append(result)
        else:
            bad.append(result)

    if len(good) < min_domain_results:
        return results

    ordered = good + bad
    for idx, result in enumerate(ordered, start=1):
        result["rank"] = idx
        result["hybrid_rank"] = idx
    return ordered


def restrict_results_by_domain(question: str, results: list[dict], min_keep: int = 5) -> list[dict]:
    domains = infer_question_domains(question)
    if not domains:
        return results

    good = [r for r in results if result_belongs_to_domains(r, domains)]
    if len(good) >= min_keep:
        return good
    return results


def filter_relevant_by_domain(question: str, relevant_docs: list[str], relevant_articles: list[str]):
    domains = infer_question_domains(question)
    if not domains:
        return relevant_docs, relevant_articles

    def ref_ok(ref: str) -> bool:
        return any(
            code in ref
            for domain in domains
            for code in DOMAIN_DOC_CODES.get(domain, [])
        )

    filtered_articles = [ref for ref in relevant_articles if ref_ok(ref)]
    if not filtered_articles:
        return relevant_docs, relevant_articles

    allowed_docs = set(ref.split("|")[0] for ref in filtered_articles)
    filtered_docs = [doc for doc in relevant_docs if doc.split("|")[0] in allowed_docs]
    return filtered_docs, filtered_articles


# ============================================================
# Question complexity / context size
# ============================================================

def is_complex_question(question: str) -> bool:
    q = question.lower()
    q_norm = q.replace("doanh nghiệp nhỏ và vừa", "dnnvv")
    q_norm = q_norm.replace("nhỏ và vừa", "dnnvv")

    complex_signals = [
        "đồng thời",
        "khác gì",
        "khác nhau",
        "so với",
        "cùng với",
        "một mặt",
        "mặt khác",
        "và nếu",
        "và khi",
        "trong trường hợp",
    ]

    if len(q_norm) >= 220:
        return True
    if any(signal in q_norm for signal in complex_signals):
        return True
    if q_norm.count(" và ") >= 3:
        return True
    return False


def choose_context_top_k(question: str, qa_module, args, search_queries: list[str] | None = None) -> int:
    search_queries = search_queries or []

    if len(search_queries) >= 3:
        return args.complex_context_top_k

    q = question.lower()
    multi_issue_markers = [
        "đồng thời",
        "vừa",
        "ngoài ra",
        "bên cạnh đó",
        "và nếu",
        "thì cần",
        "như thế nào và",
        "ra sao và",
    ]

    if any(m in q for m in multi_issue_markers):
        return args.complex_context_top_k
    if is_complex_question(question):
        return args.complex_context_top_k
    if qa_module.is_broad_question(question):
        return args.broad_context_top_k
    return args.specific_context_top_k


# ============================================================
# Query decomposition
# ============================================================

def build_retrieval_queries(question: str, max_queries: int = 5) -> list[str]:
    q = " ".join(str(question).split()).strip()
    if not q:
        return []

    q_lower = q.lower()
    queries = [q]

    complex_markers = [
        "đồng thời",
        "ngoài ra",
        "bên cạnh đó",
        "mặt khác",
        "và nếu",
        "nếu ",
        "trong trường hợp",
        "thì ",
    ]

    is_complex = (
        len(q) >= 170
        or sum(m in q_lower for m in complex_markers) >= 1
        or q.count(",") >= 2
        or q_lower.count(" và ") >= 2
    )

    if is_complex:
        parts = re.split(
            r"\bđồng thời\b|\bngoài ra\b|\bbên cạnh đó\b|\bmặt khác\b|;|\. ",
            q,
            flags=re.IGNORECASE,
        )
        for part in parts:
            part = part.strip(" ,.;:")
            if len(part) >= 35 and part not in queries:
                queries.append(part)

        chunks = re.split(r",|\bvà\b", q, flags=re.IGNORECASE)
        legal_keywords = [
            "hồ sơ",
            "thủ tục",
            "điều kiện",
            "mức phạt",
            "xử phạt",
            "khắc phục",
            "bồi thường",
            "thời hạn",
            "nghĩa vụ",
            "trách nhiệm",
            "hiệu lực",
            "chấm dứt",
            "đăng ký",
            "thông báo",
            "hóa đơn",
            "hoá đơn",
            "thuế",
            "bảo lãnh",
            "hợp đồng",
            "trọng tài",
            "dữ liệu",
            "quyền sở hữu",
        ]
        for chunk in chunks:
            chunk = chunk.strip(" ,.;:")
            if len(chunk) >= 35 and any(k in chunk.lower() for k in legal_keywords):
                if chunk not in queries:
                    queries.append(chunk)

    targeted_phrases = []
    phrase_rules = [
        (
            ["hộ kinh doanh", "lệ phí"],
            [
                "thanh toán lệ phí đăng ký hộ kinh doanh",
                "168/2025/NĐ-CP Điều 97 thanh toán lệ phí đăng ký kinh doanh",
                "đăng ký hộ kinh doanh thanh toán lệ phí qua dịch vụ thanh toán điện tử",
            ],
        ),
        (
            ["khai thiếu", "thuế"],
            [
                "125/2020/NĐ-CP xử phạt hành vi khai sai dẫn đến thiếu số tiền thuế phải nộp",
                "khai thiếu số tiền thuế phải nộp phạt 20% số tiền thuế thiếu",
                "biện pháp khắc phục nộp đủ số tiền thuế thiếu tiền chậm nộp",
            ],
        ),
        (
            ["trọng tài", "biện pháp khẩn cấp tạm thời"],
            ["54/2010/QH12 Điều 50 thủ tục áp dụng biện pháp khẩn cấp tạm thời của Hội đồng trọng tài"],
        ),
        (
            ["trọng tài", "email"],
            [
                "54/2010/QH12 Điều 16 hình thức thỏa thuận trọng tài email",
                "54/2010/QH12 Điều 18 thỏa thuận trọng tài vô hiệu",
            ],
        ),
        (
            ["quỹ đầu tư khởi nghiệp sáng tạo", "giải thể"],
            [
                "38/2018/NĐ-CP quỹ đầu tư khởi nghiệp sáng tạo giải thể quỹ thông báo hồ sơ",
                "quỹ đầu tư khởi nghiệp sáng tạo chế độ báo cáo kế toán giải thể",
            ],
        ),
        (
            ["góp vốn bằng tài sản"],
            [
                "59/2020/QH14 Điều 34 tài sản góp vốn",
                "59/2020/QH14 Điều 35 chuyển quyền sở hữu tài sản góp vốn",
                "biên bản giao nhận tài sản góp vốn",
            ],
        ),
        (
            ["tên doanh nghiệp"],
            [
                "59/2020/QH14 tên doanh nghiệp trùng gây nhầm lẫn",
                "168/2025/NĐ-CP tên doanh nghiệp trùng gây nhầm lẫn",
            ],
        ),
        (
            ["bảo hiểm tai nạn lao động", "khám sức khỏe định kỳ"],
            [
                "12/2022/NĐ-CP không đóng bảo hiểm tai nạn lao động bệnh nghề nghiệp",
                "12/2022/NĐ-CP không tổ chức khám sức khỏe định kỳ cho người lao động",
                "12/2022/NĐ-CP không thanh toán chi phí y tế cho người lao động bị tai nạn lao động",
            ],
        ),
        (
            ["hóa đơn", "khai man", "chứng từ"],
            [
                "88/2015/QH13 chứng từ kế toán căn cứ ghi sổ kế toán",
                "88/2015/QH13 hành vi bị nghiêm cấm trong kế toán khai man số liệu",
                "41/2018/NĐ-CP xử phạt vi phạm hành chính trong lĩnh vực kế toán",
            ],
        ),
    ]

    for required_terms, extra_queries in phrase_rules:
        if all(term in q_lower for term in required_terms):
            targeted_phrases.extend(extra_queries)

    for tq in targeted_phrases:
        if tq not in queries:
            queries.append(tq)

    seen = set()
    deduped = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            deduped.append(query)

    return deduped[:max_queries]


def get_result_key(result: dict) -> str:
    refs = result.get("legal_reference_keys") or result.get("legal_refs") or []
    if isinstance(refs, str):
        refs = [refs]

    parent_id = result.get("parent_article_id") or result.get("metadata", {}).get("parent_article_id")
    chunk_id = result.get("chunk_id") or result.get("metadata", {}).get("chunk_id")

    if refs:
        ref = refs[0]
        parts = ref.split("|")
        if len(parts) >= 2:
            return parts[0] + "|" + parts[1]

    if parent_id:
        return str(parent_id)
    if chunk_id:
        return str(chunk_id)
    return str(result.get("retrieval_title", ""))[:200]


def merge_ranked_results(result_lists: list[list[dict]], max_results: int = 80) -> list[dict]:
    merged = {}

    for list_idx, results in enumerate(result_lists):
        weight = 1.0 if list_idx == 0 else 0.80
        for rank, result in enumerate(results, start=1):
            key = get_result_key(result)
            score = weight / (60 + rank)
            if key not in merged:
                copied = dict(result)
                copied["_decomp_score"] = 0.0
                copied["_matched_queries"] = []
                copied["_best_source_rank"] = rank
                merged[key] = copied

            merged[key]["_decomp_score"] += score
            merged[key]["_matched_queries"].append(list_idx)
            merged[key]["_best_source_rank"] = min(merged[key].get("_best_source_rank", rank), rank)

    final_results = list(merged.values())
    final_results.sort(key=lambda x: x.get("_decomp_score", 0.0), reverse=True)

    for idx, result in enumerate(final_results, start=1):
        result["rank"] = idx
        result["hybrid_rank"] = idx
        result["hybrid_score"] = result.get("hybrid_score", result.get("_decomp_score", 0.0))

    return final_results[:max_results]


def run_decomposed_hybrid_search(
    question: str,
    hybrid_module,
    chunks,
    bm25,
    dense_model,
    dense_index,
    bm25_top_k: int,
    dense_top_k: int,
    final_top_k: int,
    max_queries: int,
):
    search_queries = build_retrieval_queries(question, max_queries=max_queries)

    result_lists = []
    for query_idx, search_query in enumerate(search_queries):
        sub_results = hybrid_module.hybrid_search(
            query=search_query,
            chunks=chunks,
            bm25=bm25,
            dense_model=dense_model,
            dense_index=dense_index,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            final_top_k=final_top_k,
        )
        sub_results = [dict(r) for r in sub_results]
        for idx, r in enumerate(sub_results, start=1):
            r["_matched_queries"] = [query_idx]
            r["_best_source_rank"] = idx
        result_lists.append(sub_results)

    if len(result_lists) == 1:
        results = result_lists[0]
        for idx, result in enumerate(results, start=1):
            result["rank"] = idx
            result["hybrid_rank"] = idx
        return results, search_queries, result_lists

    results = merge_ranked_results(result_lists=result_lists, max_results=final_top_k)
    results = restrict_results_by_domain(question=question, results=results, min_keep=8)
    return results, search_queries, result_lists


# ============================================================
# Rerank / citation candidate logic
# ============================================================

def rebalance_after_rerank(question: str, results: list[dict]) -> list[dict]:
    domains = infer_question_domains(question)
    if not results:
        return results

    for r in results:
        hybrid_rank = r.get("hybrid_rank") or r.get("rank") or 9999
        reranker_rank = r.get("reranker_rank") or 9999
        decomp_score = float(r.get("_decomp_score", 0.0))
        domain_bonus = 1.0 if result_belongs_to_domains(r, domains) else 0.0

        hybrid_score = 1.0 / max(hybrid_rank, 1)
        reranker_score = 1.0 / max(reranker_rank, 1)

        if domains:
            final_score = (
                0.20 * hybrid_score
                + 0.45 * reranker_score
                + 0.25 * domain_bonus
                + 0.10 * decomp_score
            )
        else:
            final_score = 0.35 * hybrid_score + 0.45 * reranker_score + 0.20 * decomp_score

        if domains and not result_belongs_to_domains(r, domains) and reranker_rank > 10:
            final_score *= 0.20

        r["final_context_score"] = final_score

    results = sorted(results, key=lambda x: x.get("final_context_score", 0.0), reverse=True)
    for idx, r in enumerate(results, start=1):
        r["final_context_rank"] = idx
    return results


def article_title_match_bonus(question: str, result: dict) -> float:
    q = question.lower()
    title = str(result.get("retrieval_title", "")).lower()
    refs = " ".join(result.get("legal_reference_keys", []) or []).lower()
    text = title + " " + refs

    bonus = 0.0
    phrase_groups = [
        (["hồ sơ", "đăng ký", "hộ kinh doanh"], ["hồ sơ", "đăng ký", "hộ kinh doanh"]),
        (["lệ phí", "hộ kinh doanh"], ["lệ phí", "đăng ký"]),
        (["thỏa thuận trọng tài", "vô hiệu"], ["thỏa thuận trọng tài", "vô hiệu"]),
        (["biện pháp khẩn cấp tạm thời"], ["biện pháp khẩn cấp tạm thời"]),
        (["chứng từ kế toán", "ghi sổ"], ["chứng từ kế toán", "sổ kế toán"]),
        (["khai thiếu", "thuế"], ["khai sai", "thiếu số tiền thuế"]),
        (["góp vốn bằng tài sản"], ["góp vốn", "tài sản góp vốn", "chuyển quyền sở hữu"]),
        (["tên doanh nghiệp", "gây nhầm lẫn"], ["tên doanh nghiệp", "trùng", "nhầm lẫn"]),
        (["quỹ đầu tư khởi nghiệp sáng tạo", "giải thể"], ["quỹ đầu tư khởi nghiệp sáng tạo", "giải thể"]),
    ]

    for q_terms, title_terms in phrase_groups:
        if all(term in q for term in q_terms):
            matched = sum(1 for term in title_terms if term in text)
            bonus += 0.08 * matched

    return bonus


def apply_article_title_bonus(question: str, results: list[dict]) -> list[dict]:
    for r in results:
        base = float(r.get("final_context_score", 0.0))
        r["final_context_score"] = base + article_title_match_bonus(question, r)

    results = sorted(results, key=lambda x: x.get("final_context_score", 0.0), reverse=True)
    for idx, r in enumerate(results, start=1):
        r["final_context_rank"] = idx
    return results


def dedupe_results(results: list[dict], max_results: int | None = None) -> list[dict]:
    seen = set()
    output = []
    for r in results:
        key = get_result_key(r)
        if key in seen:
            continue
        seen.add(key)
        output.append(r)
        if max_results is not None and len(output) >= max_results:
            break
    return output


def collect_citation_results(
    question: str,
    results: list[dict],
    result_lists: list[list[dict]],
    citation_top_k: int,
    per_query_citation_k: int,
) -> list[dict]:
    """
    High-recall citation mode:
    - lấy top global sau rerank
    - cộng thêm top đại diện từ từng sub-query để câu phức không mất vế.
    """
    candidates = []

    global_pool = results[:citation_top_k]
    candidates.extend(global_pool)

    domains = infer_question_domains(question)

    # Đại diện từng sub-query từ danh sách gốc trước merge, sau đó ưu tiên đúng domain.
    for q_idx, sub_results in enumerate(result_lists):
        if q_idx == 0:
            continue

        sub_results = [dict(r) for r in sub_results]
        if domains:
            good = [r for r in sub_results if result_belongs_to_domains(r, domains)]
            if len(good) >= per_query_citation_k:
                sub_results = good

        for r in sub_results[:per_query_citation_k]:
            # Nếu result này cũng nằm trong results đã rerank, lấy bản có score/rank mới.
            key = get_result_key(r)
            reranked = next((x for x in results if get_result_key(x) == key), None)
            candidates.append(reranked or r)

    candidates = dedupe_results(candidates)

    # Sort lại theo final score nếu có; các đại diện sub-query chưa rerank vẫn được giữ sau top global.
    candidates.sort(key=lambda x: float(x.get("final_context_score", x.get("final_score", 0.0))), reverse=True)
    candidates = restrict_results_by_domain(question=question, results=candidates, min_keep=3)
    candidates = dedupe_results(candidates, max_results=citation_top_k)

    return candidates


# ============================================================
# Context focusing / answer postprocessing
# ============================================================

def extract_focus_terms(question: str) -> list[str]:
    q = question.lower()
    terms = []

    phrase_map = [
        "chậm đóng bảo hiểm xã hội",
        "bảo hiểm xã hội bắt buộc",
        "giữ bản chính",
        "bằng cấp",
        "văn bằng",
        "chứng chỉ",
        "khắc phục",
        "mức hỗ trợ",
        "hỗ trợ tư vấn",
        "mức phạt",
        "xử phạt",
        "buộc",
        "trả lại",
        "điều kiện cấp bảo lãnh",
        "cấp bảo lãnh tín dụng",
        "phạt vi phạm",
        "bồi thường thiệt hại",
        "thời hạn",
        "hồ sơ",
        "thủ tục",
        "nghĩa vụ",
        "trách nhiệm",
    ]

    for phrase in phrase_map:
        if phrase in q:
            terms.append(phrase)

    for token in re.findall(r"[a-zà-ỹđ0-9]+", q):
        if len(token) >= 5 and token not in terms:
            terms.append(token)

    return terms[:24]


def focus_content_by_question(content: str, question: str, max_chars: int) -> str:
    if not content:
        return ""

    content = " ".join(str(content).split()).strip()
    if len(content) <= max_chars:
        return content

    lower_content = content.lower()
    terms = extract_focus_terms(question)

    best_pos = -1
    best_score = -1
    priority_terms = {
        "chậm đóng bảo hiểm xã hội",
        "bảo hiểm xã hội bắt buộc",
        "giữ bản chính",
        "mức hỗ trợ",
        "hỗ trợ tư vấn",
        "điều kiện cấp bảo lãnh",
        "cấp bảo lãnh tín dụng",
        "phạt vi phạm",
        "bồi thường thiệt hại",
    }

    for term in terms:
        pos = lower_content.find(term.lower())
        if pos >= 0:
            score = len(term)
            if term in priority_terms:
                score += 100
            if score > best_score:
                best_score = score
                best_pos = pos

    if best_pos < 0:
        return content[:max_chars]

    half = max_chars // 2
    start = max(0, best_pos - half)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)

    snippet = content[start:end]
    if start > 0:
        snippet = "... " + snippet
    if end < len(content):
        snippet = snippet + " ..."
    return snippet


def postprocess_answer(answer: str) -> str:
    answer = answer.strip()
    bad_phrases = [
        "Yêu cầu định dạng câu trả lời đã được đáp ứng.",
        "Yêu cầu định dạng đã được đáp ứng.",
    ]
    for phrase in bad_phrases:
        answer = answer.replace(phrase, "")
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer


def apply_answer_guard(question: str, answer: str, relevant_articles: list[str]) -> str:
    q = question.lower()
    ans = answer.strip()
    ans_lower = ans.lower()
    refs_text = "\n".join(relevant_articles)

    yes_no_markers = ["có bị", "có phải", "có được", "có cần", "được không", "phải không"]
    if any(m in q for m in yes_no_markers):
        if not ans_lower.startswith(("có", "không")):
            if any(x in ans_lower for x in ["bị phạt", "bị xử phạt", "vi phạm", "phải", "cần"]):
                ans = "Có. " + ans[0].lower() + ans[1:]
            elif any(x in ans_lower for x in ["không bắt buộc", "không phải", "không cần"]):
                ans = "Không. " + ans[0].lower() + ans[1:]

    numeric_question_markers = ["bao nhiêu", "mức phạt", "mức hỗ trợ", "tối đa", "thời hạn", "trong bao lâu", "bao lâu", "tỷ lệ", "%"]
    if any(m in q for m in numeric_question_markers):
        has_number = bool(re.search(r"\d|%|phần trăm", ans_lower))
        if not has_number:
            ans += (
                "\n\nLưu ý: Câu hỏi yêu cầu con số hoặc thời hạn cụ thể; "
                "cần đối chiếu trực tiếp căn cứ pháp lý được trích dẫn để xác định chính xác."
            )

    if any(m in q for m in ["khắc phục", "biện pháp khắc phục"]):
        if not any(m in ans_lower for m in ["khắc phục", "buộc", "trả lại", "nộp lại", "hoàn thành thủ tục"]):
            ans += (
                "\n\nNgoài hình thức xử phạt, cần xem thêm biện pháp khắc phục hậu quả "
                "được quy định tại căn cứ pháp lý tương ứng."
            )

    if (
        any(m in q for m in ["có bị phạt", "bị xử phạt", "xử lý như thế nào"])
        and "12/2022/NĐ-CP" in refs_text
        and ans_lower.startswith("không")
    ):
        ans = "Cần đối chiếu hành vi cụ thể với căn cứ xử phạt được trích dẫn. " + ans

    return ans


def ensure_full_citations_in_answer(answer: str, relevant_articles: list[str]) -> str:
    answer = answer.strip()
    markers = ["\nCăn cứ pháp lý:", "\n[Căn cứ pháp lý]"]
    for marker in markers:
        first_pos = answer.find(marker)
        if first_pos >= 0:
            answer = answer[:first_pos].strip()
            break

    if not relevant_articles:
        return answer

    refs = "\n".join(f"- {ref}" for ref in relevant_articles)
    return answer.strip() + "\n\nCăn cứ pháp lý:\n" + refs


# ============================================================
# Citation postprocess
# ============================================================

def rebuild_docs_from_articles(relevant_docs: list[str], relevant_articles: list[str], max_docs: int) -> list[str]:
    doc_map = {}
    for doc in relevant_docs:
        parts = doc.split("|", 1)
        if len(parts) == 2:
            doc_map[parts[0]] = doc

    for article in relevant_articles:
        parts = article.split("|")
        if len(parts) >= 3:
            doc_no = parts[0]
            doc_name = parts[1]
            if doc_no not in doc_map:
                doc_map[doc_no] = f"{doc_no}|{doc_name}"

    ordered_docs = []
    seen = set()
    for article in relevant_articles:
        doc_no = article.split("|")[0]
        if doc_no in doc_map and doc_no not in seen:
            ordered_docs.append(doc_map[doc_no])
            seen.add(doc_no)

    return ordered_docs[:max_docs]


def drop_low_value_amendment_refs(relevant_docs: list[str], relevant_articles: list[str]):
    if len(relevant_articles) <= 1:
        return relevant_docs, relevant_articles

    cleaned_articles = []
    for ref in relevant_articles:
        ref_lower = ref.lower()
        is_amendment_noise = (
            ("sửa đổi" in ref_lower or "bổ sung" in ref_lower)
            and (ref.endswith("|Điều 1") or ref.endswith("|Điều 4") or ref.endswith("|Điều 5"))
        )
        if is_amendment_noise and len(relevant_articles) > 1:
            continue
        cleaned_articles.append(ref)

    if not cleaned_articles:
        cleaned_articles = relevant_articles

    allowed_doc_keys = set(ref.split("|")[0] for ref in cleaned_articles)
    cleaned_docs = [doc for doc in relevant_docs if doc.split("|")[0] in allowed_doc_keys]
    if not cleaned_docs:
        cleaned_docs = relevant_docs
    return cleaned_docs, cleaned_articles


def drop_amendment_duplicates(question: str, relevant_docs: list[str], relevant_articles: list[str]):
    q = question.lower()
    if any(x in q for x in ["sửa đổi", "bổ sung", "luật sửa đổi", "văn bản sửa đổi"]):
        return relevant_docs, relevant_articles

    preferred_base_docs = [
        "50/2005/QH11", "36/2005/QH11", "45/2019/QH14", "59/2020/QH14",
        "38/2019/QH14", "123/2020/NĐ-CP", "98/2020/NĐ-CP", "125/2020/NĐ-CP",
    ]
    amendment_like_codes = ["07/2022/QH15", "70/2025/NĐ-CP", "24/2025/NĐ-CP", "102/2021/NĐ-CP"]

    has_base = any(any(code in ref for code in preferred_base_docs) for ref in relevant_articles)
    if not has_base:
        return relevant_docs, relevant_articles

    cleaned_articles = []
    for ref in relevant_articles:
        is_amendment = any(code in ref for code in amendment_like_codes)
        is_low_value_article = ref.endswith("|Điều 1") or ref.endswith("|Điều 4") or ref.endswith("|Điều 5")
        if is_amendment and is_low_value_article:
            continue

        if is_amendment:
            article = ref.split("|")[-1]
            base_same_article_exists = any(
                (base_code in other and other.endswith("|" + article))
                for base_code in preferred_base_docs
                for other in relevant_articles
            )
            if base_same_article_exists:
                continue

        cleaned_articles.append(ref)

    if not cleaned_articles:
        cleaned_articles = relevant_articles

    allowed_doc_nos = set(ref.split("|")[0] for ref in cleaned_articles)
    cleaned_docs = [doc for doc in relevant_docs if doc.split("|")[0] in allowed_doc_nos]
    if not cleaned_docs:
        cleaned_docs = relevant_docs
    return cleaned_docs, cleaned_articles


def dynamic_limit_articles_high_recall(question: str, relevant_articles: list[str], max_articles: int) -> list[str]:
    """
    High recall mode cho F2: không cắt quá ít như bản trước.
    Câu cụ thể vẫn cho tối đa 3 điều; câu broad/phức cho nhiều hơn.
    """
    q = question.lower()
    multi_markers = [
        "đồng thời", "và nếu", "ngoài ra", "bên cạnh đó", "trong trường hợp",
        "hồ sơ gồm", "trình tự", "thủ tục", "nghĩa vụ gì và", "điều kiện gì và",
        "xử lý thế nào và", "ra sao", "như thế nào",
    ]
    broad_markers = [
        "những trường hợp nào", "các trường hợp", "bao gồm những gì", "các biện pháp",
        "những nội dung gì", "quy định như thế nào", "cần thực hiện những gì",
    ]

    if any(x in q for x in multi_markers):
        return relevant_articles[:max_articles]
    if any(x in q for x in broad_markers):
        return relevant_articles[:min(max_articles, 5)]
    return relevant_articles[:min(max_articles, 3)]


def postprocess_references(question: str, relevant_docs: list[str], relevant_articles: list[str], max_docs: int, max_articles: int):
    relevant_docs, relevant_articles = filter_relevant_by_domain(question, relevant_docs, relevant_articles)
    relevant_docs = rebuild_docs_from_articles(relevant_docs, relevant_articles, max_docs=max_docs)

    relevant_docs, relevant_articles = drop_amendment_duplicates(question, relevant_docs, relevant_articles)
    relevant_docs, relevant_articles = drop_low_value_amendment_refs(relevant_docs, relevant_articles)

    relevant_articles = dynamic_limit_articles_high_recall(question, relevant_articles, max_articles=max_articles)
    relevant_docs = rebuild_docs_from_articles(relevant_docs, relevant_articles, max_docs=max_docs)
    return relevant_docs, relevant_articles


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", type=str, required=True, help="File JSON test set gồm id và question.")
    parser.add_argument("--artifact_dir", type=str, default="/kaggle/working/artifacts")
    parser.add_argument("--output_file", type=str, default="/kaggle/working/artifacts/submission.json")
    parser.add_argument("--debug_file", type=str, default="/kaggle/working/artifacts/submission_debug.json")

    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--reranker_model_name", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--load_llm_in_4bit", type=int, default=0)
    parser.add_argument("--use_reranker", type=int, default=1)

    parser.add_argument("--bm25_top_k", type=int, default=100)
    parser.add_argument("--dense_top_k", type=int, default=100)
    parser.add_argument("--candidate_top_k", type=int, default=60)

    parser.add_argument("--specific_context_top_k", type=int, default=1)
    parser.add_argument("--broad_context_top_k", type=int, default=3)
    parser.add_argument("--complex_context_top_k", type=int, default=4)

    parser.add_argument("--citation_top_k", type=int, default=8)
    parser.add_argument("--per_query_citation_k", type=int, default=2)
    parser.add_argument("--citation_max_context_chars", type=int, default=1200)
    parser.add_argument("--max_retrieval_queries", type=int, default=5)

    parser.add_argument("--hybrid_weight", type=float, default=0.70)
    parser.add_argument("--reranker_weight", type=float, default=0.30)

    parser.add_argument("--max_reranker_chars", type=int, default=3500)
    parser.add_argument("--max_context_chars", type=int, default=3000)

    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--max_docs", type=int, default=5)
    parser.add_argument("--max_articles", type=int, default=6)

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 nghĩa là chạy toàn bộ.")
    parser.add_argument("--resume", type=int, default=1, help="Nếu output_file đã có, bỏ qua các id đã xử lý.")
    parser.add_argument("--save_every", type=int, default=20)

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)
    test_file = Path(args.test_file)
    output_file = Path(args.output_file)
    debug_file = Path(args.debug_file)

    if not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy test_file: {test_file}")

    hybrid = load_module_from_path("hybrid_retrieval", root_dir / "scripts" / "04_hybrid_retrieval.py")
    qa = load_module_from_path("manual_qa_pipeline", root_dir / "scripts" / "11_run_manual_qa_eval.py")

    print("[INFO] Test file:", test_file)
    print("[INFO] Artifact dir:", artifact_dir)
    print("[INFO] Output file:", output_file)
    print("[INFO] Debug file:", debug_file)

    test_items = load_json(test_file)
    if not isinstance(test_items, list):
        raise ValueError("test_file phải là JSON list các object có id và question.")

    test_items = test_items[args.offset:]
    if args.limit and args.limit > 0:
        test_items = test_items[:args.limit]

    print("[INFO] Questions to process:", len(test_items))

    existing_outputs = {}
    if args.resume:
        existing_outputs = load_existing_outputs(output_file)
        print("[INFO] Existing outputs loaded:", len(existing_outputs))

    pending_items = [item for item in test_items if int(item["id"]) not in existing_outputs]
    print("[INFO] Pending questions:", len(pending_items))

    if not pending_items:
        final_outputs = [existing_outputs[int(item["id"])] for item in test_items]
        save_json(output_file, final_outputs)
        print("[INFO] Nothing to do. Saved existing outputs.")
        return

    bm25_path = artifact_dir / "bm25.pkl"
    chunks_path = artifact_dir / "chunks.pkl"
    dense_index_path = artifact_dir / "dense_faiss.index"

    print("[INFO] Loading BM25:", bm25_path)
    bm25 = hybrid.load_pickle(bm25_path)

    print("[INFO] Loading chunks:", chunks_path)
    chunks = hybrid.load_pickle(chunks_path)

    print("[INFO] Building lookup")
    lookup = qa.build_chunk_lookup(chunks)

    print("[INFO] Loading dense index:", dense_index_path)
    dense_index = faiss.read_index(str(dense_index_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    print("[INFO] Loading embedding model:", args.embedding_model_name)
    dense_model = SentenceTransformer(args.embedding_model_name, device=device)

    reranker = None
    if args.use_reranker:
        print("[INFO] Loading reranker:", args.reranker_model_name)
        reranker = CrossEncoder(args.reranker_model_name, max_length=512, device=device)

    prepared_items = []
    print("\n========== PHASE 1: PREPARE TEST CONTEXTS ==========")

    for idx, item in enumerate(pending_items, start=1):
        qid = int(item["id"])
        question = str(item["question"]).strip()

        # Build queries sớm để chọn context_top_k đúng với câu phức.
        search_queries_preview = build_retrieval_queries(question, max_queries=args.max_retrieval_queries)
        context_top_k = choose_context_top_k(question=question, qa_module=qa, args=args, search_queries=search_queries_preview)

        print("-" * 100)
        print(f"[{idx}/{len(pending_items)}] ID:", qid)
        print("Q:", question)
        print("Context top k:", context_top_k)

        results, search_queries, result_lists = run_decomposed_hybrid_search(
            question=question,
            hybrid_module=hybrid,
            chunks=chunks,
            bm25=bm25,
            dense_model=dense_model,
            dense_index=dense_index,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            final_top_k=args.candidate_top_k,
            max_queries=args.max_retrieval_queries,
        )

        results = prioritize_results_by_domain(question=question, results=results)

        if reranker is not None:
            results = qa.rerank_with_loaded_model(
                query=question,
                results=results,
                lookup=lookup,
                reranker=reranker,
                max_chars=args.max_reranker_chars,
            )
            results = qa.blend_ranking(results, hybrid_weight=args.hybrid_weight, reranker_weight=args.reranker_weight)
            results = rebalance_after_rerank(question=question, results=results)
            results = apply_article_title_bonus(question=question, results=results)
            results = restrict_results_by_domain(question=question, results=results, min_keep=3)
        else:
            for result in results:
                result["final_context_score"] = 1.0 / max(result.get("rank", 9999), 1)
                result["final_context_rank"] = result.get("rank")
            results = sorted(results, key=lambda x: x["final_context_score"], reverse=True)

        # Soft domain priority after rerank.
        results = prioritize_results_by_domain(question=question, results=results)

        selected = qa.deduplicate_contexts(results=results, top_k=context_top_k)
        context_blocks = qa.build_context_blocks(
            selected_results=selected,
            lookup=lookup,
            max_context_chars=args.max_context_chars,
        )
        for block in context_blocks:
            block["content"] = focus_content_by_question(
                content=block.get("content", ""),
                question=question,
                max_chars=args.max_context_chars,
            )

        # Luồng citation riêng: rộng hơn context answer, cộng thêm đại diện từng sub-query.
        citation_results = collect_citation_results(
            question=question,
            results=results,
            result_lists=result_lists,
            citation_top_k=args.citation_top_k,
            per_query_citation_k=args.per_query_citation_k,
        )
        citation_selected = qa.deduplicate_contexts(results=citation_results, top_k=args.citation_top_k)
        citation_blocks = qa.build_context_blocks(
            selected_results=citation_selected,
            lookup=lookup,
            max_context_chars=args.citation_max_context_chars,
        )

        prompt = qa.build_prompt(question=question, context_blocks=context_blocks)

        relevant_docs, relevant_articles = make_relevant_fields(
            contexts=citation_blocks,
            lookup=lookup,
            max_docs=args.max_docs,
            max_articles=args.max_articles,
        )
        relevant_docs, relevant_articles = postprocess_references(
            question=question,
            relevant_docs=relevant_docs,
            relevant_articles=relevant_articles,
            max_docs=args.max_docs,
            max_articles=args.max_articles,
        )

        if context_blocks:
            print("Search queries:", search_queries)
            print("Top context:", context_blocks[0].get("legal_reference_keys"), "|", context_blocks[0].get("retrieval_title"))
        else:
            print("Search queries:", search_queries)
            print("Top context: NONE")
        print("Relevant articles:", relevant_articles[:6])

        prepared_items.append({
            "id": qid,
            "question": question,
            "context_top_k": context_top_k,
            "search_queries": search_queries,
            "contexts": context_blocks,
            "citation_contexts": citation_blocks,
            "prompt": prompt,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })

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

    print("\n========== PHASE 2: GENERATE TEST ANSWERS ==========")
    tokenizer, llm = qa.load_llm(model_name=args.llm_model_name, load_in_4bit=bool(args.load_llm_in_4bit))

    submission_map = dict(existing_outputs)
    debug_outputs = []

    for idx, item in enumerate(prepared_items, start=1):
        qid = item["id"]
        question = item["question"]

        print("-" * 100)
        print(f"[{idx}/{len(prepared_items)}] ID:", qid)
        print("Q:", question)

        answer = qa.generate_answer_with_loaded_model(
            prompt=item["prompt"],
            tokenizer=tokenizer,
            model=llm,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        answer = postprocess_answer(answer)
        answer = apply_answer_guard(question=question, answer=answer, relevant_articles=item["relevant_articles"])
        answer = ensure_full_citations_in_answer(answer=answer, relevant_articles=item["relevant_articles"])

        submission_item = {
            "id": qid,
            "question": question,
            "answer": answer,
            "relevant_docs": item["relevant_docs"],
            "relevant_articles": item["relevant_articles"],
        }
        submission_map[qid] = submission_item

        debug_outputs.append({
            **submission_item,
            "context_top_k": item["context_top_k"],
            "search_queries": item["search_queries"],
            "contexts": item["contexts"],
            "citation_contexts": item["citation_contexts"],
        })

        print("Relevant docs:", item["relevant_docs"])
        print("Relevant articles:", item["relevant_articles"])
        print("Answer preview:", answer[:500])

        if idx % args.save_every == 0:
            final_outputs = [submission_map[int(x["id"])] for x in test_items if int(x["id"]) in submission_map]
            save_json(output_file, final_outputs)
            save_json(debug_file, {
                "processed": len(final_outputs),
                "total_requested": len(test_items),
                "debug_outputs_latest_run": debug_outputs,
            })
            print("[INFO] Checkpoint saved:", output_file)

    final_outputs = [submission_map[int(x["id"])] for x in test_items if int(x["id"]) in submission_map]
    save_json(output_file, final_outputs)
    save_json(debug_file, {
        "processed": len(final_outputs),
        "total_requested": len(test_items),
        "llm_model_name": args.llm_model_name,
        "embedding_model_name": args.embedding_model_name,
        "reranker_model_name": args.reranker_model_name if args.use_reranker else None,
        "specific_context_top_k": args.specific_context_top_k,
        "broad_context_top_k": args.broad_context_top_k,
        "complex_context_top_k": args.complex_context_top_k,
        "citation_top_k": args.citation_top_k,
        "per_query_citation_k": args.per_query_citation_k,
        "debug_outputs_latest_run": debug_outputs,
    })

    print("\n========== TESTSET INFERENCE REPORT ==========")
    print("Requested questions:", len(test_items))
    print("Generated outputs  :", len(final_outputs))
    print("Saved submission   :", output_file)
    print("Saved debug        :", debug_file)
    print("==============================================\n")


if __name__ == "__main__":
    main()
