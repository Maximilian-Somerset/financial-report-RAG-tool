from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    BM25Index,
    ChromaIndex,
    evaluate_retrieval,
    extract_document,
    generate_answer,
    reciprocal_rank_fusion,
    split_items,
)


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "data" / "report.pdf"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "models"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
)

st.set_page_config(page_title="金融报告 RAG", page_icon="📄", layout="wide")
st.title("金融报告 RAG 分析系统")
st.caption("BM25 · Chroma 向量检索 · 混合检索 · 页码引用 · 大模型回答")


@st.cache_resource(show_spinner=False)
def build_bm25(pdf_hash: str, pdf_bytes: bytes, chunk_size: int, overlap: int):
    del pdf_hash
    page_count, items = extract_document(pdf_bytes)
    chunks = split_items(items, chunk_size=chunk_size, overlap=overlap)
    return page_count, chunks, BM25Index(chunks)


@st.cache_resource(show_spinner=False)
def build_chroma(
    pdf_hash: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
    _chunks,
):
    return ChromaIndex(
        chunks=_chunks,
        pdf_hash=pdf_hash,
        persist_dir=CHROMA_DIR,
        model_name=model_name,
        model_cache_dir=MODEL_CACHE_DIR,
        chunk_size=chunk_size,
        overlap=overlap,
    )


with st.sidebar:
    st.header("系统配置")
    retrieval_mode = st.selectbox("检索模式", ["BM25", "Vector", "Hybrid"])
    chunk_size = st.slider("分块长度", 300, 1000, 600, 50)
    overlap = st.slider("重叠长度", 0, 200, 100, 20)
    top_k = st.slider("最终证据数量", 1, 8, 4)
    st.caption(f"Embedding：{EMBEDDING_MODEL}")
    if os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"):
        st.success(f"大模型已配置：{os.getenv('LLM_MODEL')}")
    else:
        st.warning("未配置大模型，将只显示检索证据。")

uploaded_file = st.file_uploader("上传 PDF 报告", type="pdf")
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name
elif DEFAULT_PDF.exists():
    pdf_bytes = DEFAULT_PDF.read_bytes()
    source_name = str(DEFAULT_PDF)
    st.info(f"正在使用服务器文件：{DEFAULT_PDF}")
else:
    st.info("请上传 PDF，或将报告保存为 data/report.pdf 后刷新页面。")
    st.stop()

pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
try:
    with st.spinner("正在解析 PDF 并建立 BM25 索引..."):
        page_count, chunks, bm25_index = build_bm25(
            pdf_hash, pdf_bytes, chunk_size, overlap
        )
except Exception as exc:
    st.error(f"PDF 解析或 BM25 建库失败：{exc}")
    st.stop()

vector_index = None
if retrieval_mode in {"Vector", "Hybrid"}:
    try:
        with st.spinner("正在加载 Embedding 模型并建立/读取 Chroma 索引..."):
            vector_index = build_chroma(
                pdf_hash, chunk_size, overlap, EMBEDDING_MODEL, chunks
            )
    except Exception as exc:
        st.error(f"向量索引初始化失败：{exc}")
        st.info("可先切换到 BM25 模式，检查模型下载和 Chroma 目录后再重试。")
        st.stop()


def search(question: str, requested_top_k: int):
    if retrieval_mode == "BM25":
        return bm25_index.search(question, requested_top_k)
    if retrieval_mode == "Vector":
        return vector_index.search(question, requested_top_k)
    candidate_k = max(requested_top_k, 6)
    return reciprocal_rank_fusion(
        [
            bm25_index.search(question, candidate_k),
            vector_index.search(question, candidate_k),
        ],
        top_k=requested_top_k,
    )


metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("PDF 页数", page_count)
metric2.metric("文本片段", len(chunks))
metric3.metric("表格片段", sum(chunk.content_type == "table" for chunk in chunks))
metric4.metric("检索模式", retrieval_mode)
st.caption(f"文件：{source_name}｜SHA256：{pdf_hash[:16]}")

qa_tab, eval_tab, inspect_tab = st.tabs(["文档问答", "检索评测", "索引检查"])

with qa_tab:
    question = st.text_input("问题")
    if st.button("检索并回答", type="primary", disabled=not question.strip()):
        try:
            results = search(question, top_k)
            answer = generate_answer(question, results)
            st.subheader("回答")
            st.markdown(answer)
            st.subheader("召回证据")
            for number, result in enumerate(results, start=1):
                with st.expander(
                    f"来源 {number}｜第 {result.chunk.page} 页｜"
                    f"{result.chunk.content_type}｜得分 {result.score:.4f}"
                ):
                    st.write(result.chunk.text)
        except Exception as exc:
            st.error(f"问答失败：{exc}")

with eval_tab:
    st.write("CSV 必须包含 question 和 expected_keyword 两列。")
    eval_file = st.file_uploader("上传评测集 CSV", type="csv", key="eval_csv")
    if eval_file is not None and st.button("运行检索评测"):
        try:
            eval_df = pd.read_csv(eval_file, encoding="utf-8-sig")
            required = {"question", "expected_keyword"}
            if not required.issubset(eval_df.columns):
                st.error("CSV 缺少 question 或 expected_keyword 列。")
            else:
                records = evaluate_retrieval(
                    search, eval_df.to_dict("records"), top_k=top_k
                )
                result_df = pd.DataFrame(records)
                st.metric(f"{retrieval_mode} Hit@{top_k}", f"{result_df['hit'].mean():.1%}")
                st.dataframe(result_df, use_container_width=True)
        except Exception as exc:
            st.error(f"评测失败：{exc}")

with inspect_tab:
    selected = st.number_input(
        "片段编号", min_value=0, max_value=max(0, len(chunks) - 1), value=0
    )
    if chunks:
        preview = chunks[int(selected)]
        st.caption(
            f"片段 {preview.chunk_id}｜PDF 第 {preview.page} 页｜{preview.content_type}"
        )
        st.text(preview.text)
