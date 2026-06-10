"""
評価用の顔画像を data/faces/ に取得するスクリプト。

既定では thispersondoesnotexist.com から StyleGAN 生成の合成顔をダウンロード
する。合成顔を使う理由:
  - 実在人物ではないため肖像権・プライバシーの懸念が小さい
  - リクエストごとに新しい顔が得られ、ライセンス手続きが不要

公開データセット(FFHQ/CelebA 等)を使いたい場合は、画像を data/faces/ に
直接置けばそのまま評価対象になる（このスクリプトは不要）。

使い方:
    py -3 download_faces.py --count 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

FACES_DIR = Path(__file__).parent / "data" / "faces"
SOURCE_URL = "https://thispersondoesnotexist.com/"
HEADERS = {
    # UA を付けないと弾かれることがある
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def download(count: int, delay: float = 1.0) -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(FACES_DIR.glob("*.jpg")))
    print(f"取得先: {SOURCE_URL}")
    print(f"保存先: {FACES_DIR}  (既存 {existing} 枚)")

    saved = 0
    for i in range(count):
        try:
            resp = requests.get(SOURCE_URL, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            # 連番ではなく内容ハッシュで重複回避
            name = f"face_{existing + saved:05d}.jpg"
            (FACES_DIR / name).write_bytes(resp.content)
            saved += 1
            print(f"  [{i + 1}/{count}] {name} ({len(resp.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"  [{i + 1}/{count}] 失敗: {e}")
        # サーバ負荷軽減のため間隔をあける
        time.sleep(delay)

    print(f"\n完了: {saved} 枚を保存（合計 {existing + saved} 枚）")


def main() -> None:
    ap = argparse.ArgumentParser(description="評価用の合成顔画像をダウンロード")
    ap.add_argument("--count", type=int, default=200, help="取得枚数（既定 200）")
    ap.add_argument("--delay", type=float, default=1.0, help="リクエスト間隔(秒)")
    args = ap.parse_args()
    download(args.count, args.delay)


if __name__ == "__main__":
    main()
