"""
顔画像評価ツール（コマンドライン版）
画像を表示して評価を入力
"""

import json
from pathlib import Path
from PIL import Image

class FaceEvaluatorCLI:
    def __init__(self, image_dir="data/faces/archive/img_align_celeba/img_align_celeba", output_file="data/ratings.json"):
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.ratings = self.load_ratings()
        self.image_files = self.get_image_files()
        self.current_index = 0
        
    def load_ratings(self):
        """既存の評価データを読み込む"""
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_ratings(self):
        """評価データを保存"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings, f, ensure_ascii=False, indent=2)
    
    def get_image_files(self):
        """画像ファイルのリストを取得"""
        if not self.image_dir.exists():
            return []
        
        extensions = ['.jpg', '.jpeg', '.png']
        files = [f for f in self.image_dir.iterdir() 
                if f.suffix.lower() in extensions]
        
        # 評価済みを除外
        return sorted([f for f in files if f.name not in self.ratings])[:50]  # 最初の50枚
    
    def run(self):
        """評価を開始"""
        if not self.image_files:
            print("評価する画像がありません")
            return
        
        print("=" * 60)
        print("顔の好み評価システム（コマンドライン版）")
        print("=" * 60)
        print(f"評価対象: {len(self.image_files)}枚")
        print("評価方法: 1-10の点数を入力（10が最高）")
        print("スキップ: sと入力、終了: qと入力\n")
        
        for i, img_path in enumerate(self.image_files, 1):
            try:
                # 画像を表示
                img = Image.open(img_path)
                img.show()
                
                print(f"\n[{i}/{len(self.image_files)}] {img_path.name}")
                
                # 評価を入力
                while True:
                    rating_input = input("評価 (1-10): ").strip().lower()
                    
                    if rating_input == 'q':
                        print("終了します")
                        self.save_ratings()
                        return
                    
                    if rating_input == 's':
                        print("スキップしました")
                        break
                    
                    try:
                        rating = int(rating_input)
                        if 1 <= rating <= 10:
                            self.ratings[img_path.name] = rating
                            self.save_ratings()
                            print(f"✓ 評価を保存しました")
                            break
                        else:
                            print("1-10の数値を入力してください")
                    except ValueError:
                        print("数値を入力してください（終了: q、スキップ: s）")
                
            except Exception as e:
                print(f"エラー: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("評価完了！")
        self.show_stats()
    
    def show_stats(self):
        """統計情報を表示"""
        if not self.ratings:
            print("評価データがありません")
            return
        
        ratings_list = list(self.ratings.values())
        avg = sum(ratings_list) / len(ratings_list)
        
        print(f"\n統計情報:")
        print(f"  評価数: {len(self.ratings)}枚")
        print(f"  平均点: {avg:.1f}")
        print(f"  最高点: {max(ratings_list)}")
        print(f"  最低点: {min(ratings_list)}")
        print(f"\n評価データ: {self.output_file}")
        print("\n次のステップ:")
        print("  python extract_features.py  # 特徴量を抽出")
        print("  python train_model.py        # モデルを訓練")

if __name__ == "__main__":
    app = FaceEvaluatorCLI()
    app.run()
