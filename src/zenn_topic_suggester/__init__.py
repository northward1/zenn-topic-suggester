import onnxruntime as ort
from transformers import AutoTokenizer
import argparse
import os
import sys
from typing import List
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from huggingface_hub import hf_hub_download
import re
import requests
import platformdirs
from pathlib import Path
import unicodedata

MODEL_ID = "intfloat/multilingual-e5-small"
ONNX_FILENAME = "onnx/model_qint8_avx512_vnni.onnx"

DATA_URL = "https://raw.githubusercontent.com/northward1/zenn-topic-suggester/refs/heads/main/data.jsonl"
CACHE_DIR = Path(platformdirs.user_cache_dir("zenn-topic-suggester"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = CACHE_DIR / "data.jsonl"


def fetch_articles_data() -> pd.DataFrame:
    # 一週間以内に取得したcacheがあるなら、それを利用する
    if os.path.isfile(DATA_PATH):
        file_mtime = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)

        if file_mtime >= datetime.now() - timedelta(days=7):
            return pd.read_json(DATA_PATH, lines=True)

    # そうでないなら、再取得する
    r = requests.get(DATA_URL)

    if r.status_code != requests.codes.ok:
        print(f"記事データのダウンロードに失敗しました。", file=sys.stderr)
        sys.exit(1)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)

    return pd.read_json(DATA_PATH, lines=True)


def search_topics(
    start: datetime = datetime.today().astimezone(ZoneInfo("Asia/Tokyo"))
    - timedelta(days=180),
    end: datetime = datetime.today().astimezone(ZoneInfo("Asia/Tokyo")),
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
    df = fetch_articles_data()

    df = df[df["published_at"].between(start, end)]

    df = df.explode("topics")

    summary = df.groupby("topics").agg(
        記事数=("authenticated_liked_count", "count"),
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


def get_east_asian_width_count(text):
    count = 0
    for c in text:
        if unicodedata.east_asian_width(c) in "FWA":
            count += 2
        else:
            count += 1
    return count


def main():
    # 引数の設定や処理
    parser = argparse.ArgumentParser(prog="zenn-topic-suggester", description="")
    parser.add_argument(
        "input_file",
        help="推奨トピックを検索したいMarkdownファイルのパスを指定します。(例: ./articles/example.md)",
    )
    parser.add_argument(
        "-s",
        "--show_detail_score",
        help="スコアの詳細なデータを出力するかどうかを指定します。デフォルトでは出力します。",
        action="store_false",
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
    parser.add_argument(
        "-ws",
        "--sim_weight",
        help="スコアを計算するときの類似度の重みを調整します。デフォルトは1.0です。",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-wc",
        "--count_weight",
        help="スコアを計算するときの記事数の重みを調整します。デフォルトは1.0です。",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-wf",
        "--found_weight",
        help="スコアを計算するときの完全一致の重みを調整します。デフォルトは0.45です。",
        type=float,
        default=0.45,
    )

    args = parser.parse_args()

    input_file = args.input_file
    lines = args.lines
    lower_bound = args.lower_bound
    show_detail_score = args.show_detail_score

    # スコア計算用の定数
    W_SIM = args.sim_weight
    W_COUNT = args.count_weight
    W_FOUND = args.found_weight

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

    regex = re.compile("|".join(map(re.escape, topics)))
    founds = set(regex.findall(content.lower()))
    df["is_found"] = df.index.isin(founds).astype(int)

    df["final_score"] = (
        min_max_scale(
            (W_SIM * df["sim_norm"])
            + (W_COUNT * df["count_norm"])
            + (W_FOUND * df["is_found"])
        )
        * 100
    )

    top_recommendations = df.sort_values(by="final_score", ascending=False).head(lines)

    for topic, row in top_recommendations.iterrows():
        if show_detail_score:
            print(
                f"- {topic + " " * (20 - get_east_asian_width_count(topic))} # スコア: {row["final_score"].round(2):6.2f}, 完全一致: {bool(row["is_found"])} 類似度: {row["sim_norm"].round(2):.2f}, 記事数: {int(row["記事数"])}"
            )
        else:
            print(f"- {topic + " " * (20 - get_east_asian_width_count(topic))}")


if __name__ == "__main__":
    main()
