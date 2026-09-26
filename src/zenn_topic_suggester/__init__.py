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


def search_topics(
    start: datetime.datetime = datetime.datetime.today().astimezone(
        ZoneInfo("Asia/Tokyo")
    )
    - datetime.timedelta(days=180),
    end: datetime.datetime = datetime.datetime.today().astimezone(
        ZoneInfo("Asia/Tokyo")
    ),
    lower_bound: int = 100,
) -> pd.DataFrame:
    """
    記事のデータをすべて読み込み、公開日が [start, end] に含まれる記事からトピックの統計情報を抽出する。
    トピックの内、紐づいた記事数が lower_bound 以上であるトピックの統計情報を返す。

    Parameters
    ----------
    start: datetime.datetime
        抽出対象とする期間の始点
    end: datetime.datetime
        抽出対象とする期間の終点
    lower_bound: int
        抽出対象とするトピックの記事数の下限

    Returns
    ----------
    summary: pd.DataFrame
        トピックの統計情報
    """
    df = pd.read_json(DATA_PATH, lines=True)

    df = df[df["published_at"].between(start, end)]

    df = df.explode("topics")

    summary = df.groupby("topics").agg(
        記事数=("authenticated_liked_count", "count"),
        平均Like数=("authenticated_liked_count", "mean"),
        Like数の中央値=("authenticated_liked_count", "median"),
    )

    # 指定した期間で lower_bound 記事以上投稿されているトピックを抽出する
    summary = summary[summary["記事数"] >= lower_bound]

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
    parser.add_argument(
        "-l",
        "--lower_bound",
        help="推奨されるトピックの候補となるトピックの期間内に投稿された記事数の下限を指定します。デフォルトは100個です。",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    input_file = args.input_file
    lines = args.lines
    lower_bound = args.lower_bound

    if not os.path.isfile(input_file):
        print(
            f"指定されたファイル {input_file} が見つかりません。パスを確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(input_file, mode="r", encoding="utf-8") as f:
        content = f.read()

    df = search_topics(lower_bound=lower_bound)

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
