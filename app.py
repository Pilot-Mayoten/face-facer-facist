"""
顔の好み診断 — MVP (ペア比較版)

2 枚の顔を並べて「どちらが好きか」をクリックで記録し、比較ログを SQLite に
貯める。十分たまったら Bradley-Terry モデルで各顔の潜在的な好み度を推定し、
「あなたの好みランキング」を表示する。

起動:
    py -3 app.py
ブラウザで http://127.0.0.1:7860 が開く。
"""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

import gradio as gr

from ranking import ranking

BASE = Path(__file__).parent
FACES_DIR = BASE / "data" / "faces"
DB_PATH = BASE / "data" / "comparisons.db"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------
# データアクセス
# --------------------------------------------------------------------------
def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS comparisons (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT NOT NULL,
                winner TEXT NOT NULL,
                loser  TEXT NOT NULL
            )
            """
        )


def record_comparison(winner: str, loser: str) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO comparisons (ts, winner, loser) VALUES (?, ?, ?)",
            (dt.datetime.now().isoformat(timespec="seconds"), winner, loser),
        )


def load_comparisons() -> list[tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT winner, loser FROM comparisons").fetchall()
    return [(w, l) for w, l in rows]


def comparison_count() -> int:
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]


# --------------------------------------------------------------------------
# 画像・ペア選択
# --------------------------------------------------------------------------
def list_faces() -> list[str]:
    if not FACES_DIR.exists():
        return []
    return sorted(
        p.name for p in FACES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


def face_path(name: str) -> str:
    return str(FACES_DIR / name)


def pick_pair(faces: list[str]) -> tuple[str, str]:
    """次に出題するペアを選ぶ。MVP ではランダム（後で能動学習に差し替え可能）。"""
    a, b = random.sample(faces, 2)
    return a, b


# --------------------------------------------------------------------------
# UI ハンドラ
# --------------------------------------------------------------------------
FACES = list_faces()


def new_pair():
    if len(FACES) < 2:
        return None, None, None, None, "data/faces/ に画像が 2 枚以上必要です。"
    a, b = pick_pair(FACES)
    status = f"比較回数: {comparison_count()}　/　顔画像: {len(FACES)} 枚"
    return face_path(a), face_path(b), a, b, status


def choose(winner_name, loser_name):
    """winner_name を勝者として記録し、次のペアを返す。"""
    if winner_name and loser_name:
        record_comparison(winner_name, loser_name)
    return new_pair()


def on_left(left_name, right_name):
    return choose(left_name, right_name)


def on_right(left_name, right_name):
    return choose(right_name, left_name)


def on_skip(left_name, right_name):
    return new_pair()


def update_ranking(top_k):
    comps = load_comparisons()
    if not comps:
        return [], "まだ比較データがありません。何回か選んでから更新してください。"
    ranked = ranking(comps, items=FACES)
    top = ranked[: int(top_k)]
    gallery = [
        (face_path(name), f"#{i + 1}  好み度 {score:.2f}")
        for i, (name, score) in enumerate(top)
    ]
    msg = f"{len(comps)} 件の比較から {len(FACES)} 枚をランキングしました。"
    return gallery, msg


# --------------------------------------------------------------------------
# 画面構築
# --------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="顔の好み診断 (ペア比較)") as demo:
        gr.Markdown(
            "# 顔の好み診断 — ペア比較\n"
            "2 枚のうち **好きな方** をクリックしてください。"
            "繰り返すほど精度が上がります。"
        )

        left_name = gr.State()
        right_name = gr.State()

        status = gr.Markdown()
        with gr.Row():
            left_img = gr.Image(label="左", height=400, show_label=False, interactive=False)
            right_img = gr.Image(label="右", height=400, show_label=False, interactive=False)
        with gr.Row():
            btn_left = gr.Button("← こっちが好き", variant="primary")
            btn_skip = gr.Button("どちらでもない / スキップ")
            btn_right = gr.Button("こっちが好き →", variant="primary")

        gr.Markdown("---\n## あなたの好みランキング")
        with gr.Row():
            top_k = gr.Slider(3, 30, value=10, step=1, label="表示数")
            btn_rank = gr.Button("ランキングを更新", variant="secondary")
        rank_msg = gr.Markdown()
        gallery = gr.Gallery(label="好きな顔トップ", columns=5, height="auto")

        outputs = [left_img, right_img, left_name, right_name, status]
        btn_left.click(on_left, [left_name, right_name], outputs)
        btn_right.click(on_right, [left_name, right_name], outputs)
        btn_skip.click(on_skip, [left_name, right_name], outputs)
        btn_rank.click(update_ranking, [top_k], [gallery, rank_msg])

        demo.load(new_pair, None, outputs)

    return demo


if __name__ == "__main__":
    init_db()
    build_app().launch()
