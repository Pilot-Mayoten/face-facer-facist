"""
新しい顔画像の好み度を予測するスクリプト（サンプル版）
"""

import json
import numpy as np
from pathlib import Path
from extract_features import FaceFeatureExtractor
from train_model import PreferenceModel

def predict_sample_images(image_dir, model_path, sample_size=100, top_n=20):
    """
    サンプル画像の好み度を予測
    
    Args:
        image_dir: 予測対象の画像ディレクトリ
        model_path: 訓練済みモデルのパス
        sample_size: 予測する画像数
        top_n: 表示する上位N枚の数
    """
    image_dir = Path(image_dir)
    
    # 既に評価済みの画像を読み込み
    ratings_file = Path("data/ratings.json")
    if ratings_file.exists():
        with open(ratings_file, 'r', encoding='utf-8') as f:
            rated_images = set(json.load(f).keys())
    else:
        rated_images = set()
    
    # 特徴量を読み込み
    features_file = Path("data/features.json")
    if not features_file.exists():
        print("特徴量ファイルが見つかりません。extract_features.pyを実行してください。")
        return
    
    with open(features_file, 'r') as f:
        features_dict = json.load(f)
    
    # モデルを読み込み
    print("モデルを読み込み中...")
    model = PreferenceModel()
    model.load_model(model_path)
    
    # 未評価の画像をサンプリング
    unrated_images = [name for name in features_dict.keys() if name not in rated_images]
    
    if not unrated_images:
        print("未評価の画像がありません")
        return
    
    # サンプルサイズを調整
    sample_size = min(sample_size, len(unrated_images))
    import random
    random.seed(42)
    sampled_images = random.sample(unrated_images, sample_size)
    
    print(f"\n{sample_size}枚の画像を予測中...")
    
    predictions = []
    
    for i, img_name in enumerate(sampled_images, 1):
        try:
            # 特徴量を取得
            embedding = np.array(features_dict[img_name])
            
            # 予測
            score = model.predict(embedding.reshape(1, -1))[0]
            predictions.append({
                'filename': img_name,
                'score': score,
                'path': str(image_dir / img_name)
            })
            
            if i % 10 == 0:
                print(f"進行中: {i}/{sample_size}", end='\r')
            
        except Exception as e:
            print(f"\nエラー ({img_name}): {e}")
            continue
    
    if not predictions:
        print("予測できる画像がありませんでした")
        return
    
    # スコアでソート
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # 結果を表示
    print("\n\n" + "="*70)
    print("🎯 予測結果 - あなたの好みの顔 TOP {}".format(top_n))
    print("="*70)
    
    for i, pred in enumerate(predictions[:top_n], 1):
        score_bar = "★" * int(pred['score']) + "☆" * (10 - int(pred['score']))
        print(f"{i:2d}. {pred['filename']:20s} | {score_bar} {pred['score']:.2f}点")
    
    print("\n" + "="*70)
    print("💔 予測結果 - 好みではない顔 BOTTOM 10")
    print("="*70)
    
    for i, pred in enumerate(predictions[-10:][::-1], 1):
        score_bar = "★" * int(pred['score']) + "☆" * (10 - int(pred['score']))
        print(f"{i:2d}. {pred['filename']:20s} | {score_bar} {pred['score']:.2f}点")
    
    # 結果を保存
    output_file = Path("data/predictions_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"\n全予測結果を {output_file} に保存しました")
    
    # 統計情報
    scores = [p['score'] for p in predictions]
    print(f"\n📊 統計情報:")
    print(f"  予測数: {len(predictions)}枚")
    print(f"  平均スコア: {np.mean(scores):.2f}")
    print(f"  標準偏差: {np.std(scores):.2f}")
    print(f"  最高スコア: {np.max(scores):.2f}")
    print(f"  最低スコア: {np.min(scores):.2f}")
    
    # 画像パスを表示
    print(f"\n💡 TOP3の画像パス:")
    for i, pred in enumerate(predictions[:3], 1):
        print(f"  {i}. {pred['path']}")

if __name__ == "__main__":
    predict_sample_images(
        image_dir="data/faces/archive/img_align_celeba/img_align_celeba",
        model_path="data/preference_model.pkl",
        sample_size=100,
        top_n=20
    )
