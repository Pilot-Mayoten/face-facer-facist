"""
顔画像を無料でダウンロードするスクリプト（改良版）
"""

import requests
import time
from pathlib import Path

def download_faces_simple(output_dir="data/faces", num_images=50):
    """
    Generated.photosの無料APIから顔画像をダウンロード
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{num_images}枚の顔画像をダウンロード中...")
    print("完全無料のAI生成画像を使用します\n")
    
    success_count = 0
    
    # 別の無料サービスを使用
    for i in range(num_images):
        try:
            # 100k-faces.com - 無料のAI生成顔データセット
            url = f"https://raw.githubusercontent.com/nndl/100k-faces/master/samples/sample_{(i % 100):04d}.jpg"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200 and len(response.content) > 1000:
                output_path = output_dir / f"face_{i+1:04d}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
                print(f"✓ ダウンロード: {success_count}/{num_images}", end='\r')
            else:
                # フォールバック: Unsplash Random Portrait
                url2 = "https://source.unsplash.com/random/512x512/?portrait,face"
                response2 = requests.get(url2, timeout=10)
                
                if response2.status_code == 200:
                    output_path = output_dir / f"face_{i+1:04d}.jpg"
                    with open(output_path, 'wb') as f:
                        f.write(response2.content)
                    success_count += 1
                    print(f"✓ ダウンロード: {success_count}/{num_images}", end='\r')
            
            time.sleep(1)  # サーバー負荷軽減
                
        except Exception as e:
            print(f"\nエラー (画像 {i+1}): {e}")
            time.sleep(2)
            continue
    
    print(f"\n\n完了: {success_count}枚の画像を {output_dir} に保存しました")
    return success_count

if __name__ == "__main__":
    import sys
    
    num = 50
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except:
            pass
    
    print("=" * 60)
    print("顔画像ダウンローダー（完全無料）")
    print("=" * 60)
    print(f"\nダウンロード枚数: {num}枚")
    print("保存先: data/faces/\n")
    
    download_faces_simple(num_images=num)
    
    print("\n次のステップ:")
    print("  python evaluate_faces.py  # 評価を開始")
