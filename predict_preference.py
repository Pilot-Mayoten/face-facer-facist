"""
新しい顔画像の好み度を予測するスクリプト
"""

import json
import numpy as np
from pathlib import Path
from extract_features import FaceFeatureExtractor
from train_model import PreferenceModel

def predict_new_images(image_dir, model_path, top_n=10):
    """
    新しい画像の好み度を予測
    
    Args:
        image_dir: 予測対象の画像ディレクトリ
        model_path: 訓練済みモデルのパス
        top_n: 表示する上位N枚の数
    """
    image_dir = Path(image_dir)
    
    # 特徴抽出器とモデルを初期化
    print("モデルを読み込み中...")
    extractor = FaceFeatureExtractor()
    model = PreferenceModel()
    model.load_model(model_path)
    
    # 画像ファイルを取得
    extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in extensions]
    
    if not image_files:
        print(f"画像が見つかりません: {image_dir}")
        return
    
    print(f"\n{len(image_files)}枚の画像を予測中...")
    
    predictions = []
    
    for img_path in image_files:
        # 特徴量を抽出
        embedding = extractor.extract_face_embedding(img_path)
        
        if embedding is not None:
            # 予測
            score = model.predict(embedding.reshape(1, -1))[0]
            predictions.append({
                'filename': img_path.name,
                'score': score,
                'path': str(img_path)
            })
    
    if not predictions:
        print("予測できる画像がありませんでした")
        return
    
    # スコアでソート
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # 結果を表示
    print("\n" + "="*60)
    print("予測結果（スコアが高いほど好み）")
    print("="*60)
    
    for i, pred in enumerate(predictions[:top_n], 1):
        print(f"{i:2d}. {pred['filename']:40s} | スコア: {pred['score']:.2f}")
    
    # 結果を保存
    output_file = Path("data/predictions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"\n全予測結果を {output_file} に保存しました")
    
    # 統計情報
    scores = [p['score'] for p in predictions]
    print(f"\n統計情報:")
    print(f"  平均スコア: {np.mean(scores):.2f}")
    print(f"  標準偏差: {np.std(scores):.2f}")
    print(f"  最高スコア: {np.max(scores):.2f}")
    print(f"  最低スコア: {np.min(scores):.2f}")

if __name__ == "__main__":
    predict_new_images(
        image_dir="data/faces/archive/img_align_celeba/img_align_celeba",
        model_path="data/preference_model.pkl",
        top_n=10
    )
