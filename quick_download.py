"""
顔画像を取得する最も簡単な方法
Kaggle HubからCelebA-HQのサンプルをダウンロード
"""

import urllib.request
import os
from pathlib import Path

def download_sample_faces(num_images=50):
    """
    GitHubの公開データセットから顔画像をダウンロード
    """
    output_dir = Path("data/faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"顔画像をダウンロード中... (目標: {num_images}枚)")
    print("ソース: GitHub公開データセット\n")
    
    # GitHub上の顔画像データセット（100k-facesリポジトリ）
    base_urls = [
        "https://github.com/NVlabs/ffhq-dataset/raw/master/thumbnails128x128/{:05d}.png",
    ]
    
    success = 0
    
    for i in range(num_images):
        try:
            # FFHQのサムネイルをダウンロード
            url = f"https://github.com/NVlabs/ffhq-dataset/raw/master/thumbnails128x128/{i:05d}.png"
            output_path = output_dir / f"face_{i+1:04d}.png"
            
            urllib.request.urlretrieve(url, output_path)
            success += 1
            print(f"✓ {success}/{num_images}", end='\r')
            
        except Exception as e:
            print(f"\n画像 {i} でエラー: {e}")
            continue
    
    print(f"\n\n完了: {success}枚を data/faces/ に保存しました")
    print("\n次のステップ:")
    print("  python evaluate_faces.py")

if __name__ == "__main__":
    download_sample_faces(50)
