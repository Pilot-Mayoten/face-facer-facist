"""
顔画像データセットをダウンロードするスクリプト
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import gdown

def download_file(url, output_path):
    """ファイルをダウンロード"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"ダウンロード完了: {output_path}")

def download_celeba_sample(output_dir="data/faces", num_images=100):
    """
    CelebAデータセットのサンプルをダウンロード
    
    注意: 完全版は1.3GBあるので、まずサンプル版で試すことを推奨
    """
    print("CelebAデータセットをダウンロード中...")
    print("注意: 完全版のダウンロードはGoogleドライブから手動で行ってください")
    print("URL: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
    print("\nサンプル画像を生成AIから取得します...")
    
    # Generated Photosのサンプルを使用
    download_generated_faces(output_dir, num_images)

def download_generated_faces(output_dir="data/faces", num_images=50):
    """
    ThisPersonDoesNotExist.com から生成顔画像をダウンロード
    実在しない人物なのでプライバシー問題なし（完全無料）
    """
    import time
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{num_images}枚の生成顔画像をダウンロード中...")
    print("これらは実在しない人物のAI生成画像です（完全無料）")
    print("少し時間がかかります...\n")
    
    success_count = 0
    
    for i in range(num_images):
        try:
            # ThisPersonDoesNotExist API（完全無料）
            url = "https://thispersondoesnotexist.com/image"
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                output_path = output_dir / f"generated_{i+1:04d}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
                print(f"✓ ダウンロード: {success_count}/{num_images}", end='\r')
            else:
                print(f"\n警告: HTTP {response.status_code}")
            
            # サーバー負荷軽減のため待機
            time.sleep(2)
                
        except Exception as e:
            print(f"\nエラー (画像 {i+1}): {e}")
            time.sleep(3)
            continue
    
    print(f"\n\n完了: {success_count}枚の画像を {output_dir} に保存しました")

def download_from_kaggle_celeba(output_dir="data/faces"):
    """
    Kaggle経由でCelebAをダウンロード
    
    事前準備:
    1. Kaggle API Tokenが必要 (https://www.kaggle.com/settings)
    2. pip install kaggle
    3. ~/.kaggle/kaggle.json に認証情報を配置
    """
    try:
        import kaggle
        print("Kaggle APIを使用してCelebAをダウンロード...")
        
        # CelebAデータセットをダウンロード
        kaggle.api.dataset_download_files(
            'jessicali9530/celeba-dataset',
            path='data/celeba',
            unzip=True
        )
        
        print("ダウンロード完了")
        
    except ImportError:
        print("Kaggle APIがインストールされていません")
        print("pip install kaggle を実行してください")
    except Exception as e:
        print(f"エラー: {e}")
        print("\nKaggle APIの設定方法:")
        print("1. https://www.kaggle.com/settings でAPI Tokenをダウンロード")
        print("2. kaggle.json を ~/.kaggle/ に配置")

def download_lfw(output_dir="data/lfw"):
    """
    LFW (Labeled Faces in the Wild) データセットをダウンロード
    約13,000枚、250MB程度
    """
    print("LFWデータセットをダウンロード中...")
    
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = output_dir / "lfw.tgz"
    
    # ダウンロード
    print("ダウンロード中... (約250MB)")
    download_file(url, tar_path)
    
    # 解凍
    print("解凍中...")
    import tarfile
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_dir)
    
    print(f"完了: {output_dir}/lfw に保存しました")
    print("このディレクトリの画像をdata/facesにコピーして使用してください")

def main():
    print("=" * 60)
    print("顔画像データセット ダウンロードツール（全て無料）")
    print("=" * 60)
    print("\n選択してください:")
    print("1. AI生成顔画像 (推奨・手軽・プライバシー配慮・完全無料)")
    print("2. LFW データセット (13,000枚・250MB・無料)")
    print("3. CelebA (Kaggle経由・要API設定・無料)")
    print("4. 手動ダウンロードの案内（無料）")
    
    choice = input("\n選択 (1-4): ").strip()
    
    if choice == "1":
        num = input("ダウンロード枚数 (推奨: 50-100): ").strip()
        num = int(num) if num.isdigit() else 50
        download_generated_faces(num_images=num)
        
    elif choice == "2":
        download_lfw()
        
    elif choice == "3":
        download_from_kaggle_celeba()
        
    elif choice == "4":
        print("\n=== 手動ダウンロードの方法 ===")
        print("\n【CelebA - 大規模・多様性高い】")
        print("1. Google Drive: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
        print("2. img_align_celeba.zip をダウンロード (1.3GB)")
        print("3. data/faces/ に解凍")
        
        print("\n【FFHQ - 高品質】")
        print("1. https://github.com/NVlabs/ffhq-dataset")
        print("2. thumbnails128x128.zip をダウンロード")
        print("3. data/faces/ に解凍")
        
        print("\n【日本人特化データセット】")
        print("1. Asian Face Dataset (GitHub検索)")
        print("2. または、ぱくたそ・photoACから手動収集")
    
    else:
        print("無効な選択です")

if __name__ == "__main__":
    main()
