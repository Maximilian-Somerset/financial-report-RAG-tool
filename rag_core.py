from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import fitz
import jieba
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_SCHEMA_VERSION = "v2-table"


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page: int
    content_type: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ExtractedItem:
    page: int
    content_type: str
    text: str


def _overlap_ratio(block_bbox, table_bbox) -> float:
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    width = max(0.0, min(bx1, tx1) - max(bx0, tx0))
    height = max(0.0, min(by1, ty1) - max(by0, ty0))
    intersection = width * height
    block_area = max((bx1 - bx0) * (by1 - by0), 1.0)
    return intersection / block_area


def _escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip().replace("|", "\\|")


def _table_to_markdown(rows: list[list[object]]) -> str:
    cleaned = [[_escape_markdown_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    column_count = max(len(row) for row in cleaned)
    normalized = [row + [""] * (column_count - len(row)) for row in cleaned]
    header = normalized[0]
    if not any(header):
        header = [f"列{i}" for i in range(1, column_count + 1)]
    separator = ["---"] * column_count

    def markdown_row(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    return "\n".join(
        [markdown_row(header), markdown_row(separator)]
        + [markdown_row(row) for row in normalized[1:]]
    )


def extract_document(pdf_bytes: bytes) -> tuple[int, list[ExtractedItem]]:
    """Extract normal text and tables separately while retaining PDF page numbers."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        items: list[ExtractedItem] = []
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            try:
                found_tables = list(page.find_tables().tables)
            except Exception:
                found_tables = []

            table_bboxes = [tuple(table.bbox) for table in found_tables]
            text_parts: list[str] = []
            for block in page.get_text("blocks"):
                if len(block) < 7 or int(block[6]) != 0:
                    continue
                bbox = tuple(float(value) for value in block[:4])
                if any(_overlap_ratio(bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes):
                    continue
                text = str(block[4]).strip()
                if text:
                    text_parts.append(text)

            page_text = "\n".join(text_parts).strip()
            if page_text:
                items.append(ExtractedItem(page_number, "text", page_text))

            for table_number, table in enumerate(found_tables, start=1):
                try:
                    markdown = _table_to_markdown(table.extract())
                except Exception:
                    markdown = ""
                if markdown:
                    items.append(
                        ExtractedItem(
                            page_number,
                            "table",
                            f"表格 {table_number}\n{markdown}",
                        )
                    )
        return page_count, items
    finally:
        document.close()


def split_items(
    items: Iterable[ExtractedItem], chunk_size: int = 600, overlap: int = 100
) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for item in items:
        text = re.sub(r"[ \t]+", " ", item.text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        for start in range(0, len(text), step):
            part = text[start : start + chunk_size].strip()
            if len(part) < 30:
                continue
            chunks.append(
                Chunk(len(chunks), item.page, item.content_type, part)
            )
            if start + chunk_size >= len(text):
                break
    return chunks


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%]+", " ", text.lower())
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("PDF 未提取出可检索文本")
        self.chunks = chunks
        self.model = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, question: str, top_k: int = 4) -> list[SearchResult]:
        scores = self.model.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(self.chunks[i], float(scores[i])) for i in order[:top_k]
        ]


class ChromaIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        pdf_hash: str,
        persist_dir: str,
        model_name: str,
        model_cache_dir: str,
        chunk_size: int,
        overlap: int,
    ):
        if not chunks:
            raise ValueError("PDF 未提取出可检索文本")

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer(
            model_name,
            cache_folder=model_cache_dir,
            device="cpu",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
        collection_name = (
            f"report-{INDEX_SCHEMA_VERSION}-{pdf_hash[:12]}-"
            f"{chunk_size}-{overlap}-{model_hash}"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() != len(chunks):
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._add_chunks(chunks)

    def _add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.collection.add(
                ids=[str(chunk.chunk_id) for chunk in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        "page": chunk.page,
                        "chunk_id": chunk.chunk_id,
                        "content_type": chunk.content_type,
                    }
                    for chunk in batch
                ],
            )

    def search(self, question: str, top_k: int = 4) -> list[SearchResult]:
        query_vector = self.encoder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        response = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]

        results: list[SearchResult] = []
        for text, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            if text is None or metadata is None:
                continue
            chunk = Chunk(
                chunk_id=int(metadata["chunk_id"]),
                page=int(metadata["page"]),
                content_type=str(metadata.get("content_type", "text")),
                text=str(text),
            )
            results.append(SearchResult(chunk, 1.0 - float(distance)))
        return results


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], top_k: int = 6, rank_constant: int = 60
) -> list[SearchResult]:
    fused_scores: dict[int, float] = {}
    chunk_by_id: dict[int, Chunk] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            chunk_by_id[chunk_id] = result.chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                rank_constant + rank
            )

    ordered = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    return [SearchResult(chunk_by_id[i], fused_scores[i]) for i in ordered]


def evidence_only_answer(results: list[SearchResult]) -> str:
    if not results:
        return "没有检索到相关内容。"
    lines = ["当前未配置大模型，以下为检索到的原文证据："]
    for number, result in enumerate(results, start=1):
        lines.append(
            f"\n[来源{number}｜第{result.chunk.page}页｜{result.chunk.content_type}]\n"
            f"{result.chunk.text}"
        )
    return "\n".join(lines)


def generate_answer(
    question: str,
    results: list[SearchResult],
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL")
    if not api_key or not model:
        return evidence_only_answer(results)

    evidence = "\n\n".join(
        f"[来源{i}｜第{result.chunk.page}页｜{result.chunk.content_type}]\n"
        f"{result.chunk.text}"
        for i, result in enumerate(results, start=1)
    )
    system_prompt = (
        "你是金融文档分析助手。只能依据用户提供的证据回答，不得补充证据之外的事实。"
        "所有数字和关键结论后必须标注已有的[来源N｜第N页]。"
        "如果证据不足，明确回答无法从当前文档确认。回答使用中文，保持简洁。"
    )
    client_options: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索证据：\n{evidence}",
            },
        ],
    )
    return response.choices[0].message.content or "模型未返回内容。"


def evaluate_retrieval(
    search_function,
    rows: Iterable[dict[str, str]],
    top_k: int = 4,
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        question = str(row["question"]).strip()
        keyword = str(row["expected_keyword"]).strip()
        results = search_function(question, top_k)
        retrieved_text = "\n".join(result.chunk.text for result in results)
        output.append(
            {
                "question": question,
                "expected_keyword": keyword,
                "hit": keyword.lower() in retrieved_text.lower(),
                "pages": ", ".join(str(result.chunk.page) for result in results),
            }
        )
    return output
