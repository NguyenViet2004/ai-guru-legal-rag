#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Article-level Doc-first v2 inference.
- Builds no post-hoc manual fixes: all decisions happen inside this pipeline.
- Resume-safe, frequent checkpoints, explicit progress by ID.
"""
import os, sys, json, pickle, re, unicodedata, time, argparse, zipfile, math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


def norm_text(s):
    s = str(s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

STOP = set("va cua la co cho khi neu thi de duoc trong voi ve cac nhung mot hai ba bon nam sau tren duoi theo nay do vao ra hoac dong thoi phai can cong ty doanh nghiep nguoi ben viec noi dung quy dinh truong hop".split())
def tokenize(s):
    toks = norm_text(s).split()
    return [t for t in toks if len(t)>1 and t not in STOP]

DOMAIN_RULES = [
    ("sme_support", ["doanh nghiep nho va vua", "dnvv", "dnnvv", "ho tro doanh nghiep", "sieu nho", "khoi nghiep sang tao", "chuoi gia tri", "cum lien ket", "tu van vien", "dao tao", "mat bang san xuat"],
     ["04/2017/QH14", "80/2021/NĐ-CP", "06/2022/TT-BKHDT", "52/2023/TT-BTC", "39/2019/NĐ-CP", "34/2018/NĐ-CP", "38/2018/NĐ-CP"]),
    ("startup_fund", ["quy dau tu khoi nghiep", "quy dau tu khoi nghiep sang tao", "dau tu khoi nghiep sang tao", "cung dau tu"],
     ["38/2018/NĐ-CP", "04/2017/QH14", "80/2021/NĐ-CP"]),
    ("business_reg", ["dang ky doanh nghiep", "dang ky kinh doanh", "thanh lap cong ty", "ten doanh nghiep", "tru so chinh", "chi nhanh", "giai the", "tam ngung", "ho kinh doanh", "giay chung nhan dang ky"],
     ["168/2025/NĐ-CP", "59/2020/QH14"]),
    ("corporate", ["cong ty co phan", "dai hoi dong co dong", "hoi dong thanh vien", "von dieu le", "gop von", "chuyen nhuong von", "thanh vien moi", "nguoi dai dien"],
     ["59/2020/QH14", "168/2025/NĐ-CP"]),
    ("tax", ["thue", "khai thue", "quan ly thue", "mien thue", "giam thue", "hoan thue", "thu nhap doanh nghiep", "tieu thu dac biet", "nop thue", "xu phat thue"],
     ["38/2019/QH14", "80/2021/TT-BTC", "126/2020/NĐ-CP", "125/2020/NĐ-CP", "67/2025/QH15", "320/2025/NĐ-CP"]),
    ("labor", ["lao dong", "nhan vien", "hop dong lao dong", "thu viec", "tien luong", "bao hiem xa hoi", "tai nan lao dong", "benh nghe nghiep", "an toan ve sinh lao dong", "cong doan", "dinh cong", "thoa uoc lao dong"],
     ["45/2019/QH14", "12/2022/NĐ-CP", "145/2020/NĐ-CP", "84/2015/QH13", "28/2021/TT-BLĐTBXH", "50/2024/QH15"]),
    ("accounting", ["ke toan", "bao cao tai chinh", "kiem toan", "chung tu ke toan", "so ke toan", "doanh nghiep sieu nho"],
     ["88/2015/QH13", "133/2016/TT-BTC", "132/2018/TT-BTC", "41/2018/NĐ-CP"]),
    ("accounting_penalty", ["xu phat ke toan", "vi pham ke toan", "phat ke toan", "che tai ke toan", "xu phat kiem toan", "vi pham kiem toan"],
     ["41/2018/NĐ-CP", "88/2015/QH13"]),
    ("commercial", ["thuong mai", "mua ban hang hoa", "dai ly thuong mai", "logistics", "nhuong quyen thuong mai", "khuyen mai", "hoi cho", "trien lam", "van chuyen hang hoa", "quang cao thuong mai"],
     ["36/2005/QH11"]),
    ("civil", ["dan su", "giao dich dan su", "hop dong", "vo hieu", "nang luc hanh vi", "tai san bao dam", "cam co", "the chap", "bao lanh", "xu ly tai san"],
     ["91/2015/QH13", "21/2021/NĐ-CP"]),
    ("arbitration", ["trong tai", "thoa thuan trong tai", "ban trong tai", "hoi dong trong tai", "to tung trong tai"],
     ["54/2010/QH12"]),
    ("consumer", ["nguoi tieu dung", "khach hang", "hop dong theo mau", "dieu kien giao dich chung", "bao ve thong tin", "du lieu khach hang", "xoa du lieu", "mien trach nhiem"],
     ["19/2023/QH15", "55/2024/NĐ-CP"]),
    ("ip", ["so huu tri tue", "sang che", "nhan hieu", "chi dan dia ly", "thanh dinh noi dung", "don dang ky"],
     ["50/2005/QH11"]),
]

GENERIC_TITLES = ["pham vi", "doi tuong ap dung", "giai thich tu ngu", "nguyen tac", "chinh sach", "trach nhiem", "hieu luc", "dieu khoan thi hanh"]
ALLOW_GENERIC_Q = ["pham vi", "doi tuong", "giai thich", "khai niem", "nguyen tac", "tieu chi", "dieu kien", "trach nhiem", "hieu luc", "nguon von", "can cu"]


def detect_domains(q):
    nq = norm_text(q)
    docs = set(); names=set()
    for name, kws, docnums in DOMAIN_RULES:
        if any(kw in nq for kw in kws):
            names.add(name); docs.update(docnums)
    return names, docs


def split_subqueries(q, max_sub=4):
    q = str(q or "").strip()
    parts = [q]
    # Split multi-issue legal questions but keep meaningful clauses.
    seps = [" đồng thời ", ";", " và nếu ", " nếu ", " trong trường hợp ", " thì ", " và "]
    tmp=[q]
    for sep in seps:
        new=[]
        for x in tmp:
            new += [p.strip(" ,.;:-") for p in x.split(sep) if len(p.strip())>=25]
        tmp = new if len(new)>1 else tmp
    # Prefer clauses with legal-action words.
    scored=[]
    for p in [q]+tmp:
        np_ = norm_text(p)
        if len(np_)<18: continue
        score = len(np_.split())
        for kw in ["xu phat", "khac phuc", "ho so", "dieu kien", "nghia vu", "quyen", "thoi han", "thu tuc", "dang ky", "mien giam", "trach nhiem"]:
            if kw in np_: score += 10
        scored.append((score,p))
    out=[]; seen=set()
    for _,p in sorted(scored, reverse=True):
        k=norm_text(p)
        if k not in seen:
            out.append(p); seen.add(k)
        if len(out)>=max_sub: break
    return out or [q]


def complexity(q):
    nq = norm_text(q)
    c = 0
    for kw in ["dong thoi", "va neu", "nhung", "cac", "bao gom", "nhu the nao va", "quy trinh", "ho so", "nghia vu", "quyen", "xu ly", "khac phuc", "muc xu phat", "thoi han"]:
        if kw in nq: c += 1
    # Multiple legal domains often require multiple articles.
    domains,_ = detect_domains(q)
    if len(domains) >= 2: c += 1
    return c


def title_overlap_score(q_tokens, title_tokens):
    if not q_tokens or not title_tokens:
        return 0.0
    qs=set(q_tokens); ts=set(title_tokens)
    inter=len(qs & ts)
    return min(1.0, inter / max(3, min(len(qs), len(ts))))


def is_generic_article(rec, q):
    art_int = rec.get("article_number_int", 9999)
    title = norm_text(rec.get("article_title", "") + " " + rec.get("article_label", ""))
    nq = norm_text(q)
    if any(k in nq for k in ALLOW_GENERIC_Q):
        return False
    if art_int <= 5 and any(k in title for k in GENERIC_TITLES):
        return True
    return False


def answer_template(question, selected):
    refs = [r["article_ref"] for r in selected]
    lines = ["Dựa trên các căn cứ pháp lý truy xuất được, cần đối chiếu trực tiếp các quy định dưới đây để xác định quyền, nghĩa vụ, điều kiện hoặc chế tài áp dụng cho tình huống được hỏi.", "", "Căn cứ pháp lý:"]
    for ref in refs:
        lines.append(f"- {ref}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--debug_file", required=True)
    ap.add_argument("--embedding_model_name", default="BAAI/bge-m3")
    ap.add_argument("--reranker_model_name", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=2)
    ap.add_argument("--dense_top_k", type=int, default=90)
    ap.add_argument("--bm25_top_k", type=int, default=90)
    ap.add_argument("--prefilter_top_k", type=int, default=64)
    ap.add_argument("--rerank_top_k", type=int, default=48)
    ap.add_argument("--max_subqueries", type=int, default=4)
    args = ap.parse_args()

    import torch, faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder

    art_dir = Path(args.artifact_dir)
    with open(art_dir/"article_records.pkl", "rb") as f:
        records = pickle.load(f)
    with open(art_dir/"article_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    index = faiss.read_index(str(art_dir/"article_faiss.index"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device} records={len(records)} offset={args.offset} limit={args.limit}", flush=True)
    emb_model = SentenceTransformer(args.embedding_model_name, device=device)
    reranker = CrossEncoder(args.reranker_model_name, device=device, max_length=512)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test = json.load(f)
    subset = test[args.offset: args.offset + args.limit]

    out_path=Path(args.output_file); dbg_path=Path(args.debug_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outputs=[]; debugs=[]; done_ids=set()
    if args.resume and out_path.exists():
        try:
            outputs=json.load(open(out_path,"r",encoding="utf-8"))
            done_ids={int(x["id"]) for x in outputs}
            print(f"[resume] loaded outputs={len(outputs)} last_id={max(done_ids) if done_ids else None}", flush=True)
        except Exception as e:
            print("[resume] failed output",repr(e),flush=True)
    if args.resume and dbg_path.exists():
        try:
            debugs=json.load(open(dbg_path,"r",encoding="utf-8"))
        except Exception:
            debugs=[]

    def save():
        outputs_sorted=sorted(outputs,key=lambda x:int(x["id"]))
        json.dump(outputs_sorted, open(out_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(debugs, open(dbg_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    def retrieve_candidates(question):
        subqs = split_subqueries(question, args.max_subqueries)
        q_tokens = tokenize(question)
        domain_names, domain_docs = detect_domains(question)
        cand = {}
        def add(idx, dense=0.0, bm=0.0, source=""):
            if idx < 0 or idx >= len(records): return
            d = cand.setdefault(idx, {"dense":0.0,"bm25":0.0,"sources":set()})
            d["dense"] = max(d["dense"], float(dense))
            d["bm25"] = max(d["bm25"], float(bm))
            if source: d["sources"].add(source)

        for qi, sq in enumerate(subqs):
            q_emb = emb_model.encode([sq], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D,I = index.search(q_emb, min(args.dense_top_k, len(records)))
            for rank,(idx,score) in enumerate(zip(I[0],D[0])):
                add(int(idx), dense=float(score), source=f"dense{qi}")
            toks=tokenize(sq)
            scores=bm25.get_scores(toks)
            if len(scores):
                topn=min(args.bm25_top_k, len(scores))
                top_idx=np.argpartition(scores, -topn)[-topn:]
                maxs=float(scores[top_idx].max()) if top_idx.size else 1.0
                if maxs <= 0: maxs=1.0
                for idx in top_idx:
                    add(int(idx), bm=float(scores[idx]/maxs), source=f"bm25{qi}")

        # prefilter score, cheap features
        scored=[]
        for idx,d in cand.items():
            r=records[idx]
            ttl=title_overlap_score(q_tokens, r.get("title_tokens") or tokenize(r.get("title_text","")))
            doc_bonus=0.0
            if r.get("doc_number") in domain_docs:
                doc_bonus=1.0
            # do not hard-penalize cross-domain; multi-domain questions exist.
            generic_pen = 0.10 if is_generic_article(r, question) else 0.0
            pre = 0.45*d["dense"] + 0.35*d["bm25"] + 0.12*ttl + 0.08*doc_bonus - generic_pen
            scored.append((pre, idx, ttl, doc_bonus, generic_pen, d))
        scored.sort(reverse=True, key=lambda x:x[0])
        return scored[:args.prefilter_top_k], subqs, sorted(domain_names), sorted(domain_docs)

    def infer_one(item):
        q = item.get("question") or item.get("query") or item.get("prompt") or ""
        candidates, subqs, domain_names, domain_docs = retrieve_candidates(q)
        if not candidates:
            return {"id":item["id"],"question":q,"answer":"Không tìm thấy căn cứ pháp lý phù hợp.","relevant_docs":[],"relevant_articles":[]}, {"id":item["id"],"error":"no candidates"}

        rerank_items = candidates[:args.rerank_top_k]
        pairs=[(q, records[idx]["retrieval_text"][:3000]) for _,idx,_,_,_,_ in rerank_items]
        try:
            rr=np.asarray(reranker.predict(pairs, batch_size=16, show_progress_bar=False), dtype="float32")
        except TypeError:
            rr=np.asarray(reranker.predict(pairs, batch_size=16), dtype="float32")
        if len(rr)>0:
            rr_min=float(rr.min()); rr_max=float(rr.max())
            rr_norm=(rr-rr_min)/(rr_max-rr_min+1e-6)
        else:
            rr_norm=rr

        q_tokens=tokenize(q)
        final=[]
        for j,(pre, idx, ttl, doc_bonus, generic_pen, d) in enumerate(rerank_items):
            r=records[idx]
            dense=float(d["dense"]); bm=float(d["bm25"])
            rrs=float(rr_norm[j]) if j < len(rr_norm) else 0.0
            # More weight on reranker but preserve lexical/title/domain evidence.
            score=0.58*rrs + 0.18*dense + 0.12*bm + 0.08*ttl + 0.06*doc_bonus - generic_pen
            final.append({"idx":idx,"rec":r,"score":score,"rerank":float(rr[j]) if j<len(rr) else None,"rerank_norm":rrs,"dense":dense,"bm25":bm,"title":ttl,"doc_bonus":doc_bonus,"generic_pen":generic_pen,"sources":list(d["sources"])})
        final.sort(key=lambda x:x["score"], reverse=True)

        comp=complexity(q)
        nq=norm_text(q)
        if comp <= 0:
            max_articles=2; max_docs=2; rel=0.84; per_doc=2
        elif comp == 1:
            max_articles=3; max_docs=3; rel=0.78; per_doc=2
        elif comp == 2:
            max_articles=4; max_docs=3; rel=0.73; per_doc=2
        else:
            max_articles=5; max_docs=4; rel=0.68; per_doc=3
        # Some broad support questions legitimately need 4 articles, but avoid 6+.
        if any(k in nq for k in ["nhung noi dung", "bao gom nhung", "nhung chi phi", "nhung loai", "nhung hinh thuc"]):
            max_articles=max(max_articles,4); max_docs=max(max_docs,3); rel=min(rel,0.74)

        selected=[]; seen_art=set(); doc_counts=Counter(); selected_docs=[]; seen_docs=set()
        top_score=final[0]["score"] if final else 0
        for it in final:
            r=it["rec"]
            if r["article_ref"] in seen_art:
                continue
            if len(selected) >= max_articles:
                break
            if r["doc_ref"] not in seen_docs and len(seen_docs) >= max_docs:
                continue
            if doc_counts[r["doc_ref"]] >= per_doc:
                continue
            # Relative score gate. Always take top1, then score-based additions.
            if selected and it["score"] < top_score * rel:
                # Allow very strong domain/title matches for complex questions only.
                if not (comp >= 2 and it["doc_bonus"] > 0 and it["title"] >= 0.25 and it["score"] >= top_score*0.62):
                    continue
            # If generic and already have a specific article from same doc, skip unless it is high.
            if is_generic_article(r, q) and doc_counts[r["doc_ref"]] > 0 and it["score"] < top_score*0.90:
                continue
            selected.append(r)
            seen_art.add(r["article_ref"])
            doc_counts[r["doc_ref"]] += 1
            if r["doc_ref"] not in seen_docs:
                seen_docs.add(r["doc_ref"]); selected_docs.append(r["doc_ref"])

        # Minimal fallback: if only one article for complex question, add next high-confidence different aspect.
        if comp >= 2 and len(selected) < 2:
            for it in final:
                r=it["rec"]
                if r["article_ref"] in seen_art: continue
                if len(selected) >= min(3,max_articles): break
                if r["doc_ref"] not in seen_docs and len(seen_docs) >= max_docs: continue
                if it["score"] >= top_score*0.60:
                    selected.append(r); seen_art.add(r["article_ref"]); doc_counts[r["doc_ref"]]+=1
                    if r["doc_ref"] not in seen_docs:
                        seen_docs.add(r["doc_ref"]); selected_docs.append(r["doc_ref"])

        relevant_articles=[r["article_ref"] for r in selected]
        relevant_docs=[]; sd=set()
        for r in selected:
            if r["doc_ref"] not in sd:
                relevant_docs.append(r["doc_ref"]); sd.add(r["doc_ref"])
        ans=answer_template(q, selected)
        out={"id":item["id"],"question":q,"answer":ans,"relevant_docs":relevant_docs,"relevant_articles":relevant_articles}
        dbg={
            "id":item["id"],"question":q,"complexity":comp,"subqueries":subqs,"domains":domain_names,"domain_docs":domain_docs,
            "selected":[{"article_ref":r["article_ref"],"doc_ref":r["doc_ref"],"article_title":r.get("article_title") } for r in selected],
            "top_candidates":[{"ref":it["rec"]["article_ref"],"score":round(it["score"],4),"rr":round(it["rerank_norm"],4),"dense":round(it["dense"],4),"bm25":round(it["bm25"],4),"title":round(it["title"],4),"doc_bonus":it["doc_bonus"],"generic_pen":it["generic_pen"]} for it in final[:10]],
        }
        return out, dbg

    start=time.time(); processed=0
    total=len(subset)
    for local_i,item in enumerate(subset, start=1):
        iid=int(item["id"])
        if iid in done_ids:
            continue
        try:
            out,dbg=infer_one(item)
        except Exception as e:
            print(f"[error] id={iid} {repr(e)}", flush=True)
            out={"id":iid,"question":item.get("question",""),"answer":"Không tìm thấy căn cứ pháp lý phù hợp.","relevant_docs":[],"relevant_articles":[]}
            dbg={"id":iid,"error":repr(e)}
        outputs.append(out); debugs.append(dbg); done_ids.add(iid); processed+=1
        if processed % 1 == 0:
            elapsed=time.time()-start
            print(f"[progress] offset={args.offset} done_new={processed} total_done={len(done_ids)}/{total} id={iid} elapsed={elapsed/60:.1f}m arts={len(out['relevant_articles'])} docs={len(out['relevant_docs'])}", flush=True)
        if processed % args.save_every == 0:
            save()
    save()
    print(f"[finished] offset={args.offset} outputs={len(outputs)} elapsed={(time.time()-start)/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
