import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

import onnxruntime as ort
from transformers import AutoTokenizer
import argparse
import sys
from typing import List, Optional
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from huggingface_hub import hf_hub_download
import requests
import platformdirs
from pathlib import Path
import unicodedata
import ahocorasick
import sqlite3
import gzip
import shutil

MODEL_ID = "intfloat/multilingual-e5-small"
ONNX_FILENAME = "onnx/model_qint8_avx512_vnni.onnx"

DATA_URL = "https://github.com/northward1/zenn-stats-crawler/releases/latest/download/zenn_stats.db.gz"
CACHE_DIR = Path(platformdirs.user_cache_dir("zenn-topic-suggester"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "zenn_stats.db"


def fetch_database(reload_cache: bool = False) -> Path:
    """
    zenn-stats-crawler の最新データベースを取得してキャッシュに保存する。
    1週間以内に取得したキャッシュが存在する場合はそれを利用する。
    """
    if not reload_cache and DB_PATH.is_file():
        file_mtime = datetime.fromtimestamp(DB_PATH.stat().st_mtime)
        if file_mtime >= datetime.now() - timedelta(days=7):
            return DB_PATH

    print("記事データベースをダウンロードしています...", file=sys.stderr)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) zenn-topic-suggester"
    }

    tmp_path = CACHE_DIR / "zenn_stats.db.tmp"
    try:
        with requests.get(DATA_URL, headers=headers, stream=True) as r:
            r.raise_for_status()
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(gz, f_out)
        tmp_path.replace(DB_PATH)
        print("記事データベースのダウンロードが完了しました。", file=sys.stderr)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"記事データベースのダウンロードに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    return DB_PATH


def fetch_articles_data(reload_cache: bool = False) -> Path:
    """後方互換性のためのエイリアス"""
    return fetch_database(reload_cache)


def search_topics(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    lower_bound: int = 100,
    reload_cache: bool = False,
) -> pd.DataFrame:
    """
    記事のデータベースを読み込み、公開日が [start, end] に含まれる記事からトピックの統計情報を抽出する。
    トピックの内、紐づいた記事数が lower_bound 以上であるトピックの統計情報を返す。

    Parameters
    ----------
    start: datetime.datetime | None
        抽出対象とする期間の始点（デフォルト: 現在から180日前）
    end: datetime.datetime | None
        抽出対象とする期間の終点（デフォルト: 現在）
    lower_bound: int
        抽出対象とするトピックの記事数の下限
    reload_cache: bool
        キャッシュを強制更新するかどうか

    Returns
    ----------
    summary: pd.DataFrame
        トピックの統計情報（インデックス: topics, カラム: 記事数）
    """
    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.today().astimezone(jst)

    if start is None:
        start = now_jst - timedelta(days=180)
    elif start.tzinfo is None:
        start = start.replace(tzinfo=jst)
    else:
        start = start.astimezone(jst)

    if end is None:
        end = now_jst
    elif end.tzinfo is None:
        end = end.replace(tzinfo=jst)
    else:
        end = end.astimezone(jst)

    db_file = fetch_database(reload_cache)

    query = """
        SELECT j.value AS topics, COUNT(DISTINCT a.url_hash) AS 記事数
        FROM articles a, json_each(a.topics) j
        WHERE a.status = 'success'
          AND a.published_at >= ?
          AND a.published_at <= ?
        GROUP BY j.value
        HAVING COUNT(DISTINCT a.url_hash) >= ?
        ORDER BY 記事数 DESC
    """

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(start.isoformat(), end.isoformat(), lower_bound),
            index_col="topics",
        )

    return df


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
        "-r",
        "--reload_cache",
        help="最新の記事データに更新したいときに指定します。デフォルトでは前回の取得から一週間以上経過したときだけ更新します。",
        action="store_true",
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
    reload_cache = args.reload_cache

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

    df = search_topics(lower_bound=lower_bound, reload_cache=reload_cache)

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

    automaton = ahocorasick.Automaton()

    for topic in topics:
        automaton.add_word(topic, topic)

    automaton.make_automaton()
    founds = {found_word for _, found_word in automaton.iter(content.lower())}

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
