"""
Local web UI for the Chroma RAG assistant.

Usage:
    python rag_ui.py

Open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import chroma_rag


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


HTML_PAGE = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chroma RAG Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --border: #d9e0ea;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --danger: #b42318;
      --code: #f0f4f8;
      --radius: 8px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-width: 320px;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      letter-spacing: 0;
    }

    button,
    textarea,
    input {
      font: inherit;
    }

    button {
      border: 0;
      cursor: pointer;
    }

    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      padding: 24px;
      border-right: 1px solid var(--border);
      background: #fbfcfe;
    }

    .brand {
      margin-bottom: 24px;
    }

    .brand h1 {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
    }

    .brand p,
    .meta,
    .file-list {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }

    .panel {
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--border);
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
    }

    .meta-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid #edf1f6;
    }

    .file-list {
      margin: 0;
      padding-left: 18px;
    }

    .secondary {
      width: 100%;
      min-height: 40px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
    }

    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }

    .topbar {
      padding: 22px 28px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }

    .topbar h2 {
      margin: 0 0 6px;
      font-size: 20px;
    }

    .status {
      color: var(--muted);
      font-size: 14px;
    }

    .messages {
      padding: 24px 28px;
      overflow: auto;
    }

    .message {
      max-width: 880px;
      margin: 0 0 16px;
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel);
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .message.user {
      margin-left: auto;
      border-color: #b8d8d4;
      background: #edf8f6;
    }

    .role {
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .contexts {
      margin-top: 12px;
      padding: 12px;
      border-radius: var(--radius);
      background: var(--code);
      color: #344054;
      font-size: 13px;
      white-space: normal;
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      padding: 18px 28px 24px;
      border-top: 1px solid var(--border);
      background: var(--panel);
    }

    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px;
      color: var(--text);
    }

    .send {
      min-width: 92px;
      border-radius: var(--radius);
      background: var(--accent);
      color: white;
      font-weight: 700;
    }

    .send:disabled,
    .secondary:disabled {
      cursor: wait;
      opacity: 0.62;
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 820px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      .main {
        min-height: 70vh;
      }

      .composer {
        grid-template-columns: 1fr;
      }

      .send {
        min-height: 44px;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <h1>Chroma RAG Assistant</h1>
        <p>OpenAI Embeddings、ChromaDB、Responses API を使ったローカル知識ベース質問応答サンプルです。</p>
      </div>

      <section class="panel">
        <h2>索引ステータス</h2>
        <div class="meta">
          <div class="meta-row"><span>Collection</span><strong id="collectionName">-</strong></div>
          <div class="meta-row"><span>Chunks</span><strong id="collectionCount">-</strong></div>
          <div class="meta-row"><span>GPT モデル</span><strong id="modelName">-</strong></div>
          <div class="meta-row"><span>Embedding</span><strong id="embeddingName">-</strong></div>
        </div>
      </section>

      <section class="panel">
        <h2>知識ベースファイル</h2>
        <ul id="fileList" class="file-list"></ul>
      </section>

      <section class="panel">
        <button id="rebuildButton" class="secondary" type="button">ベクトル索引を再構築</button>
      </section>
    </aside>

    <main class="main">
      <header class="topbar">
        <h2>知識ベース質問応答</h2>
        <div id="statusLine" class="status">ローカルサービスに接続しています...</div>
      </header>

      <section id="messages" class="messages" aria-live="polite"></section>

      <form id="askForm" class="composer">
        <textarea id="questionInput" rows="2" placeholder="質問を入力してください。例: 年次有給休暇は何日前までに申請が必要ですか？"></textarea>
        <button id="sendButton" class="send" type="submit">送信</button>
      </form>
    </main>
  </div>

  <script>
    const state = {
      busy: false,
      messages: [
        {
          role: "assistant",
          text: "こんにちは。knowledge/ ディレクトリ内の資料に基づいて回答します。"
        }
      ]
    };

    const nodes = {
      collectionName: document.querySelector("#collectionName"),
      collectionCount: document.querySelector("#collectionCount"),
      modelName: document.querySelector("#modelName"),
      embeddingName: document.querySelector("#embeddingName"),
      fileList: document.querySelector("#fileList"),
      statusLine: document.querySelector("#statusLine"),
      messages: document.querySelector("#messages"),
      askForm: document.querySelector("#askForm"),
      questionInput: document.querySelector("#questionInput"),
      sendButton: document.querySelector("#sendButton"),
      rebuildButton: document.querySelector("#rebuildButton")
    };

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function setBusy(isBusy, label) {
      state.busy = isBusy;
      nodes.sendButton.disabled = isBusy;
      nodes.rebuildButton.disabled = isBusy;
      nodes.statusLine.textContent = label;
    }

    function renderMessages() {
      nodes.messages.innerHTML = state.messages.map((message) => {
        const contextHtml = message.contexts && message.contexts.length
          ? `<div class="contexts"><strong>参照チャンク</strong><br>${message.contexts.map((item) =>
              `[${item.number}] ${escapeHtml(item.source)}#chunk-${item.chunk_index}`
            ).join("<br>")}</div>`
          : "";
        return `<article class="message ${message.role}">
          <div class="role">${message.role === "user" ? "あなた" : "アシスタント"}</div>
          <div>${escapeHtml(message.text)}</div>
          ${contextHtml}
        </article>`;
      }).join("");
      nodes.messages.scrollTop = nodes.messages.scrollHeight;
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "リクエストに失敗しました");
      }
      return data;
    }

    async function refreshStatus() {
      try {
        const data = await requestJson("/api/status");
        nodes.collectionName.textContent = data.collection_name;
        nodes.collectionCount.textContent = data.collection_count;
        nodes.modelName.textContent = data.model;
        nodes.embeddingName.textContent = data.embedding_model;
        nodes.fileList.innerHTML = data.files.map((item) =>
          `<li>${escapeHtml(item.source)} (${item.characters} chars)</li>`
        ).join("");
        nodes.statusLine.textContent = "接続済み";
      } catch (error) {
        nodes.statusLine.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
    }

    async function ask(question) {
      const trimmed = question.trim();
      if (!trimmed || state.busy) {
        return;
      }

      state.messages.push({role: "user", text: trimmed});
      const pending = {role: "assistant", text: "回答を生成しています..."};
      state.messages.push(pending);
      nodes.questionInput.value = "";
      renderMessages();
      setBusy(true, "検索して回答を生成しています...");

      try {
        const data = await requestJson("/api/ask", {
          method: "POST",
          body: JSON.stringify({question: trimmed, top_k: 3})
        });
        pending.text = data.answer;
        pending.contexts = data.contexts;
        nodes.collectionCount.textContent = data.collection_count;
        renderMessages();
        setBusy(false, "接続済み");
      } catch (error) {
        pending.text = `エラーが発生しました: ${error.message}`;
        renderMessages();
        setBusy(false, "リクエスト失敗");
      }
    }

    async function rebuild() {
      if (state.busy) {
        return;
      }
      setBusy(true, "索引を再構築しています...");
      try {
        await requestJson("/api/rebuild", {method: "POST", body: "{}"});
        await refreshStatus();
        setBusy(false, "索引を再構築しました");
      } catch (error) {
        nodes.statusLine.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
        setBusy(false, "再構築に失敗しました");
      }
    }

    nodes.askForm.addEventListener("submit", (event) => {
      event.preventDefault();
      ask(nodes.questionInput.value);
    });

    nodes.questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask(nodes.questionInput.value);
      }
    });

    nodes.rebuildButton.addEventListener("click", rebuild);

    renderMessages();
    refreshStatus();
  </script>
</body>
</html>
"""


class RagWebApp:
    def __init__(self) -> None:
        self.openai_client = None
        self.collection = None
        self.lock = threading.Lock()
        self.model = os.getenv("RAG_MODEL", chroma_rag.DEFAULT_MODEL)
        self.embedding_model = os.getenv(
            "RAG_EMBEDDING_MODEL",
            chroma_rag.DEFAULT_EMBEDDING_MODEL,
        )

    def get_openai_client(self):
        if self.openai_client is None:
            self.openai_client = chroma_rag.create_openai_client()
        return self.openai_client

    def get_collection(self, rebuild: bool = False):
        if self.collection is None or rebuild:
            self.collection = chroma_rag.get_chroma_collection(rebuild=rebuild)
        return self.collection

    def status(self) -> dict[str, Any]:
        with self.lock:
            collection_count = self.ensure_index(rebuild=False)
            files = [
                {"source": source, "characters": len(text)}
                for source, text in chroma_rag.read_knowledge_files(chroma_rag.KNOWLEDGE_DIR)
            ]
            return {
                "collection_name": chroma_rag.COLLECTION_NAME,
                "collection_count": collection_count,
                "files": files,
                "model": self.model,
                "embedding_model": self.embedding_model,
            }

    def ensure_index(self, rebuild: bool = False) -> int:
        collection = self.get_collection(rebuild=rebuild)
        expected_count = chroma_rag.expected_chunk_count(chroma_rag.KNOWLEDGE_DIR)
        expected_signature = chroma_rag.knowledge_signature(chroma_rag.KNOWLEDGE_DIR)
        manifest = chroma_rag.read_index_manifest()
        if (
            rebuild
            or collection.count() != expected_count
            or manifest.get("knowledge_signature") != expected_signature
        ):
            collection = self.get_collection(rebuild=True)
            chroma_rag.index_knowledge(
                collection=collection,
                openai_client=self.get_openai_client(),
                knowledge_dir=chroma_rag.KNOWLEDGE_DIR,
                embedding_model=self.embedding_model,
            )
        return collection.count()

    def rebuild(self) -> dict[str, Any]:
        with self.lock:
            count = self.ensure_index(rebuild=True)
            return {
                "collection_name": chroma_rag.COLLECTION_NAME,
                "collection_count": count,
                "embedding_model": self.embedding_model,
            }

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("質問を入力してください。")

        model = str(payload.get("model") or self.model)
        top_k = int(payload.get("top_k") or 3)
        top_k = max(1, min(top_k, 8))

        with self.lock:
            self.ensure_index(rebuild=False)
            collection = self.get_collection()
            contexts = chroma_rag.retrieve(
                collection=collection,
                openai_client=self.get_openai_client(),
                question=question,
                embedding_model=self.embedding_model,
                top_k=top_k,
            )
            prompt = chroma_rag.build_prompt(question, contexts)
            answer = chroma_rag.gpt_generate(self.get_openai_client(), prompt, model=model)
            return {
                "answer": answer,
                "contexts": contexts,
                "collection_count": collection.count(),
            }


APP = RagWebApp()


class RagRequestHandler(BaseHTTPRequestHandler):
    server_version = "RagUi/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rag-ui] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_html(HTML_PAGE)
            return
        if self.path == "/api/status":
            self.send_json(APP.status())
            return
        self.send_error_json(404, "Not found")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/ask":
                self.send_json(APP.answer(self.read_json()))
                return
            if self.path == "/api/rebuild":
                self.read_json()
                self.send_json(APP.rebuild())
                return
            self.send_error_json(404, "Not found")
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(500, str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_bytes(200, "text/html; charset=utf-8", data)

    def send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(200, "application/json; charset=utf-8", data)

    def send_error_json(self, status_code: int, message: str) -> None:
        data = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status_code, "application/json; charset=utf-8", data)

    def send_bytes(self, status_code: int, content_type: str, data: bytes) -> None:
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True


def main() -> None:
    chroma_rag.load_env_files()
    parser = argparse.ArgumentParser(description="Local web UI for the Chroma RAG assistant")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RagRequestHandler)
    print(f"RAG UI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
