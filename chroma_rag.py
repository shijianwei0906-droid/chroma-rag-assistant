"""
Chroma + OpenAI Embeddings + GPT RAG example.

Usage:
    python chroma_rag.py
    python chroma_rag.py "年次有給休暇は何日前までに申請が必要ですか？"
    python chroma_rag.py "経費精算は何営業日で振り込まれますか？" --show-prompt

Rebuild vector database:
    python chroma_rag.py --rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "_chroma_data"
INDEX_MANIFEST_PATH = CHROMA_DIR / "knowledge_manifest.json"
COLLECTION_NAME = "company_handbook"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_EXTENSIONS = {".txt", ".md"}
STOP_CHARS = set("のはがをにでとやもへからまでですますするしたしているあるこれそれどれ誰何いつどこ？。、,.!? ")

SYSTEM_PROMPT = """あなたは社内ナレッジベースの質問応答アシスタントです。
必ず提示された資料だけに基づいて回答し、推測で補完しないでください。
資料に答えがない場合は「分かりません」と回答してください。
回答は簡潔にし、最後に参照した資料番号を記載してください。
"""


class MockOpenAIClient:
    """Local OpenAI-compatible stub used when no API key is configured."""

    embedding_dimensions = 64

    def __init__(self) -> None:
        self.embeddings = SimpleNamespace(create=self._create_embeddings)
        self.responses = SimpleNamespace(create=self._create_response)
        self.is_mock = True

    def _create_embeddings(self, **kwargs):
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            SimpleNamespace(embedding=self._embed_text(str(text)))
            for text in inputs
        ]
        return SimpleNamespace(data=data)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.embedding_dimensions
        tokens = tokenize_for_rerank(text)
        for token in tokens or [text]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                vector[index % self.embedding_dimensions] += (byte / 255.0) - 0.5
        magnitude = sum(value * value for value in vector) ** 0.5
        if not magnitude:
            return vector
        return [value / magnitude for value in vector]

    def _create_response(self, **kwargs):
        prompt = str(kwargs.get("input", ""))
        answer = (
            "[mock-openai] OPENAI_API_KEY is not set, so this is a local simulated "
            "answer. Chroma retrieval and prompt construction still ran; configure "
            "OPENAI_API_KEY in .env to get a real model response."
        )
        if prompt:
            answer += "\n\nContext preview:\n" + prompt[:700]
        return SimpleNamespace(output_text=answer)


@dataclass
class Chunk:
    id: str
    source: str
    chunk_index: int
    text: str


def load_env_file(path: Path) -> None:
    """Load a .env file without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_files() -> None:
    load_env_file(BASE_DIR / ".env")


def create_openai_client():
    load_env_files()
    if os.getenv("RAG_MOCK_OPENAI", "").lower() in TRUTHY_VALUES:
        return MockOpenAIClient()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("gpt")
    if not api_key:
        return MockOpenAIClient()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Package 'openai' is missing. Run: python -m pip install openai") from exc

    return OpenAI(api_key=api_key)


def get_chroma_collection(rebuild: bool = False):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Package 'chromadb' is missing. Run: python -m pip install chromadb") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def read_knowledge_files(knowledge_dir: Path) -> list[tuple[str, str]]:
    if not knowledge_dir.exists():
        raise RuntimeError(f"Knowledge directory does not exist: {knowledge_dir}")

    files = [
        path
        for path in sorted(knowledge_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise RuntimeError(f"No .txt or .md files found in knowledge directory: {knowledge_dir}")

    documents = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            source = str(path.relative_to(BASE_DIR)).replace("\\", "/")
            documents.append((source, text))
    return documents


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks


def split_text(text: str, chunk_size: int = 350, overlap: int = 60) -> list[str]:
    """Split markdown by sections first, then by character windows."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc_title = next((line for line in lines if line.startswith("# ")), "")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("# "):
            continue

        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = []

        if doc_title and not current:
            current.append(doc_title)
        current.append(line)

    if current:
        sections.append("\n".join(current))

    if not sections:
        sections = ["\n".join(lines)]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(split_long_text(section, chunk_size, overlap))

    return chunks


def make_chunk_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return f"{source}:{chunk_index}:{digest[:12]}"


def build_chunks(knowledge_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in read_knowledge_files(knowledge_dir):
        for index, chunk_text in enumerate(split_text(text), start=1):
            chunks.append(
                Chunk(
                    id=make_chunk_id(source, index, chunk_text),
                    source=source,
                    chunk_index=index,
                    text=chunk_text,
                )
            )
    return chunks


def expected_chunk_count(knowledge_dir: Path = KNOWLEDGE_DIR) -> int:
    return len(build_chunks(knowledge_dir))


def knowledge_signature(knowledge_dir: Path = KNOWLEDGE_DIR) -> str:
    digest = hashlib.sha1()
    for source, text in read_knowledge_files(knowledge_dir):
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_index_manifest() -> dict[str, Any]:
    if not INDEX_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_index_manifest(collection_count: int, signature: str) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "collection_count": collection_count,
                "knowledge_signature": signature,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def embed_texts(client: Any, texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def tokenize_for_rerank(text: str) -> set[str]:
    chinese_chars = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {token for token in chinese_chars + english_words if token not in STOP_CHARS}


def keyword_score(question: str, text: str) -> float:
    question_tokens = tokenize_for_rerank(question)
    if not question_tokens:
        return 0.0
    text_tokens = tokenize_for_rerank(text)
    return len(question_tokens & text_tokens) / len(question_tokens)


def index_knowledge(
    collection,
    openai_client: Any,
    knowledge_dir: Path,
    embedding_model: str,
) -> int:
    chunks = build_chunks(knowledge_dir)
    if not chunks:
        raise RuntimeError("No text chunks available for indexing.")

    embeddings = embed_texts(openai_client, [chunk.text for chunk in chunks], embedding_model)
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {"source": chunk.source, "chunk_index": chunk.chunk_index}
            for chunk in chunks
        ],
    )
    write_index_manifest(len(chunks), knowledge_signature(knowledge_dir))
    return len(chunks)


def retrieve(
    collection,
    openai_client: Any,
    question: str,
    embedding_model: str,
    top_k: int,
) -> list[dict[str, Any]]:
    question_embedding = embed_texts(openai_client, [question], embedding_model)[0]
    candidate_count = max(top_k, min(max(top_k * 4, top_k), collection.count()))
    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    contexts = []
    for text, metadata, distance in zip(docs, metadatas, distances):
        lexical = keyword_score(question, text)
        rerank_score = lexical * 1.8 - float(distance)
        contexts.append(
            {
                "number": 0,
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
                "keyword_score": lexical,
                "rerank_score": rerank_score,
            }
        )
    contexts.sort(key=lambda item: item["rerank_score"], reverse=True)
    contexts = contexts[:top_k]
    for index, item in enumerate(contexts, start=1):
        item["number"] = index
    return contexts


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        "\n".join(
            [
                f"[{item['number']}] 出典: {item['source']}#chunk-{item['chunk_index']}",
                f"内容: {item['text']}",
            ]
        )
        for item in contexts
    )

    return f"""以下の資料だけに基づいて質問に回答してください。資料に答えがない場合は「分かりません」と回答してください。

資料:
{context_text}

質問:
{question}
"""


def gpt_generate(openai_client: Any, prompt: str, model: str) -> str:
    response = openai_client.responses.create(
        model=model,
        temperature=0.05,
        store=False,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    return response.output_text.strip()


def ask(
    question: str,
    model: str,
    embedding_model: str,
    top_k: int,
    rebuild: bool,
    show_prompt: bool,
) -> None:
    openai_client = create_openai_client()
    collection = get_chroma_collection(rebuild=rebuild)

    manifest = read_index_manifest()
    if (
        not rebuild
        and (
            collection.count() != expected_chunk_count(KNOWLEDGE_DIR)
            or manifest.get("knowledge_signature") != knowledge_signature(KNOWLEDGE_DIR)
        )
    ):
        collection = get_chroma_collection(rebuild=True)

    if rebuild or collection.count() == 0:
        chunk_count = index_knowledge(
            collection=collection,
            openai_client=openai_client,
            knowledge_dir=KNOWLEDGE_DIR,
            embedding_model=embedding_model,
        )
        print(f"Knowledge base indexed: {chunk_count} chunks")

    contexts = retrieve(
        collection=collection,
        openai_client=openai_client,
        question=question,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    prompt = build_prompt(question, contexts)
    answer = gpt_generate(openai_client, prompt, model=model)

    print("User question:")
    print(question)
    print("\nChroma results:")
    for item in contexts:
        print(
            f"- [{item['number']}] {item['source']}#chunk-{item['chunk_index']} "
            f"(distance={item['distance']:.4f})"
        )
    if show_prompt:
        print("\nPrompt sent to GPT:")
        print(prompt)
    print("\nModel:")
    print(model)
    print("\nEmbedding model:")
    print(embedding_model)
    print("\nAnswer:")
    print(answer)


def main() -> None:
    load_env_files()
    parser = argparse.ArgumentParser(description="Chroma + GPT RAG example")
    parser.add_argument("question", nargs="?", default="試用期間中の従業員は年次有給休暇を取得できますか？")
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the Chroma vector database")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt sent to GPT")
    args = parser.parse_args()

    ask(
        question=args.question,
        model=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        rebuild=args.rebuild,
        show_prompt=args.show_prompt,
    )


if __name__ == "__main__":
    main()
