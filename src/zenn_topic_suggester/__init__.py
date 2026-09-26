import onnxruntime as ort
from transformers import AutoTokenizer
import argparse
import os
import sys
from typing import List
from pathlib import Path
import numpy as np
import datetime
import pandas as pd
from zoneinfo import ZoneInfo
from huggingface_hub import hf_hub_download

MODEL_ID = "intfloat/multilingual-e5-small"
ONNX_FILENAME = "onnx/model_qint8_avx512_vnni.onnx"

DATA_PATH = Path(__file__).resolve().parent / "data.jsonl"


def search_topics() -> pd.DataFrame:
    df = pd.read_json(DATA_PATH, lines=True)

    # 半年前までの記事のデータを抽出する
    now = datetime.datetime.today().astimezone(ZoneInfo("Asia/Tokyo"))
    half_a_year_ago = now - datetime.timedelta(days=180)

    df = df[df["published_at"] >= half_a_year_ago]

    df = df.explode("topics")

    summary = df.groupby("topics").agg(
        記事数=("authenticated_liked_count", "count"),
        平均Like数=("authenticated_liked_count", "mean"),
        Like数の中央値=("authenticated_liked_count", "median"),
    )

    # 半年間で100記事以上投稿されているトピックを抽出する
    summary = summary[summary["記事数"] >= 100]

    return summary


def get_embeddings(
    texts: List[str],
    tokenizer,
    session,
    prefix: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    all_embeddings = []
    model_inputs = [node.name for node in session.get_inputs()]

    prefixed_texts = [f"{prefix}{text}" for text in texts] if prefix else texts

    for i in range(0, len(prefixed_texts), batch_size):
        batch_texts = prefixed_texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        onnx_inputs = {}
        for name in model_inputs:
            if name in inputs:
                onnx_inputs[name] = inputs[name]
            elif name == "token_type_ids":
                onnx_inputs[name] = np.zeros_like(inputs["input_ids"])

        outputs = session.run(None, onnx_inputs)

        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]

        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def min_max_scale(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def main():
    # 引数の設定や処理
    parser = argparse.ArgumentParser(prog="zenn-topic-suggester", description="")
    parser.add_argument(
        "input_file",
        help="推奨トピックを検索したいMarkdownファイルのパスを指定します。(例: ./articles/example.md)",
    )
    parser.add_argument(
        "-n",
        "--lines",
        help="表示する推奨トピックの数を指定します。デフォルトは20個です。",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    input_file = args.input_file
    lines = args.lines

    if not os.path.isfile(input_file):
        print(
            f"指定されたファイル {input_file} が見つかりません。パスを確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(input_file, mode="r", encoding="utf-8") as f:
        content = f.read()

    df = search_topics()

    topics = df.index.values

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        onnx_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=ONNX_FILENAME,
        )
        session = ort.InferenceSession(onnx_path)
    except Exception as e:
        print(f"モデルのロード・ダウンロードに失敗しました。: {e}", file=sys.stderr)
        sys.exit(1)

    content_vec = get_embeddings([content], tokenizer, session, prefix="query: ")[0]
    topic_vecs = get_embeddings(
        topics,
        tokenizer,
        session,
        prefix="passage: ",
        batch_size=128,
    )

    scores = np.dot(topic_vecs, content_vec)

    df["similarity"] = scores
    df["sim_norm"] = min_max_scale(df["similarity"])

    df["count_log"] = np.log1p(df["記事数"])
    df["count_norm"] = min_max_scale(df["count_log"])

    W_SIM = 0.90
    W_COUNT = 0.10

    df["final_score"] = (W_SIM * df["sim_norm"]) + (W_COUNT * df["count_norm"])

    top_recommendations = df.sort_values(by="final_score", ascending=False).head(lines)

    for topic, row in top_recommendations.iterrows():
        print(
            f'- {topic:<20} # スコア: {row["final_score"].round(2):.2f}, 類似度: {row["sim_norm"].round(2):.2f}, 直近半年の記事数: {int(row["記事数"])}'
        )


if __name__ == "__main__":
    main()
