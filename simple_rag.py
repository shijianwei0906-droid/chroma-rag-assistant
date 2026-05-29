"""
Minimal RAG example using in-memory documents and OpenAI Responses API.

Usage:
    python simple_rag.py "年次有給休暇は何日前までに申請が必要ですか？"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_MODEL = "gpt-4.1-mini"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
STOP_CHARS = set("のはがをにでとやもへからまでですますするしたしているあるこれそれどれ誰何いつどこ？。、,.!? ")
SYSTEM_PROMPT = """あなたは社内ナレッジベースの質問応答アシスタントです。
必ず提示された資料だけに基づいて回答し、推測で補完しないでください。
資料に答えがない場合は「分かりません」と回答してください。
回答は簡潔にし、根拠となる資料もできるだけ示してください。
"""


class MockOpenAIClient:
    """Small local stand-in used when no API key is configured."""

    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create_response)

    def _create_response(self, **kwargs):
        prompt = str(kwargs.get("input", ""))
        answer = (
            "[mock-openai] OPENAI_API_KEY is not set, so this is a local simulated "
            "answer. The retrieval pipeline still ran; configure OPENAI_API_KEY "
            "in .env to get a real model response."
        )
        if prompt:
            answer += "\n\nContext preview:\n" + prompt[:500]
        return SimpleNamespace(output_text=answer)


@dataclass
class Document:
    source: str
    text: str


DOCS = [
    Document(
        source="従業員ハンドブック-年次有給休暇",
        text="正社員は毎年 10 日の年次有給休暇を取得できます。試用期間中の従業員は年次有給休暇の対象外です。",
    ),
    Document(
        source="従業員ハンドブック-申請フロー",
        text="年次有給休暇は分割して取得できますが、少なくとも 3 日前までにシステムで申請する必要があります。",
    ),
    Document(
        source="従業員ハンドブック-病気休暇",
        text="病気休暇を取得する場合は当日中に直属上司へ連絡してください。2 日以上連続する場合は医療機関の証明書が必要です。",
    ),
]


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
    load_env_file(Path(__file__).with_name(".env"))


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


def tokenize(text: str) -> set[str]:
    """Tokenize Chinese characters and alphanumeric words for simple keyword matching."""
    chinese_chars = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {token for token in chinese_chars + english_words if token not in STOP_CHARS}


def retrieve(question: str, docs: list[Document], top_k: int = 2) -> list[Document]:
    question_tokens = tokenize(question)

    scored_docs = []
    for doc in docs:
        doc_tokens = tokenize(doc.text)
        overlap = question_tokens & doc_tokens
        score = len(overlap) / max(len(question_tokens), 1)
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k] if score > 0]


def build_prompt(question: str, context_docs: list[Document]) -> str:
    context = "\n".join(
        f"[{index}] 出典: {doc.source}\n内容: {doc.text}"
        for index, doc in enumerate(context_docs, start=1)
    )

    return f"""以下の資料だけに基づいて質問に回答してください。資料に答えがない場合は「分かりません」と回答してください。

資料:
{context}

質問:
{question}
"""


def gpt_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    client = create_openai_client()
    response = client.responses.create(
        model=model,
        temperature=0.05,
        store=False,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    return response.output_text.strip()


def ask(question: str, model: str = DEFAULT_MODEL) -> None:
    context_docs = retrieve(question, DOCS)
    prompt = build_prompt(question, context_docs)
    answer = gpt_generate(prompt, model=model)

    print("User question:")
    print(question)
    print("\nRetrieved context:")
    for doc in context_docs:
        print(f"- {doc.source}: {doc.text}")
    print("\nPrompt sent to GPT:")
    print(prompt)
    print("Model:")
    print(model)
    print("\nAnswer:")
    print(answer)


def main() -> None:
    load_env_files()
    parser = argparse.ArgumentParser(description="Minimal RAG + GPT example")
    parser.add_argument("question", nargs="?", default="試用期間中の従業員は年次有給休暇を取得できますか？")
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()
    ask(args.question, model=args.model)


if __name__ == "__main__":
    main()
