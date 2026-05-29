# chroma-rag-assistant

ChromaDB と OpenAI API を使った、ローカル実行用の RAG 質問応答サンプルです。コマンドラインでの質問応答、Chroma によるベクトル検索、軽量な Web UI を含みます。

## 機能

- `knowledge/` ディレクトリ内の `.md` / `.txt` ファイルを読み込み
- OpenAI Embeddings を使って ChromaDB のローカルベクトル索引を作成
- OpenAI Responses API で検索結果に基づく回答を生成
- コマンドラインとブラウザ Web UI の両方に対応
- ローカルベクトルデータベースの再構築に対応

## 必要環境

- Python 3.10+
- OpenAI API Key

## インストール

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 設定

環境変数ファイルの例をコピーします。

```bash
copy .env.example .env
```

`.env` に以下を設定します。

```env
OPENAI_API_KEY=your_openai_api_key_here
```

古い変数名 `gpt` にも対応していますが、標準の `OPENAI_API_KEY` を推奨します。

## 使い方

コマンドライン RAG:

```bash
python chroma_rag.py "年次有給休暇は何日前までに申請が必要ですか？"
```

ChromaDB 索引の再構築:

```bash
python chroma_rag.py --rebuild
```

Web UI の起動:

```bash
python rag_ui.py
```

ブラウザで以下を開きます。

```text
http://127.0.0.1:8000
```

## プロジェクト構成

```text
.
├── chroma_rag.py       # ChromaDB + OpenAI RAG の主処理
├── rag_ui.py           # ローカル Web UI
├── simple_rag.py       # ChromaDB を使わない最小 RAG サンプル
├── knowledge/          # サンプル知識ベース
├── requirements.txt
└── .env.example
```

## 注意

`_chroma_data/`、`.env`、`__pycache__/` などのローカル生成ファイルは Git にコミットしません。GitHub にアップロードする前に、`.env` に実際の API Key が含まれていないことを確認してください。
