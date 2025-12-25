"""
顔画像から特徴量を抽出するモジュール
FaceNet (facenet-pytorch) を使用して顔の特徴ベクトルを抽出
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceFeatureExtractor:
    def __init__(self, device=None):
        """
        顔特徴抽出器を初期化
        
        Args:
            device: 使用するデバイス (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"使用デバイス: {self.device}")
        
        # 顔検出モデル (MTCNN)
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            device=self.device,
            post_process=False
        )
        
        # 顔特徴抽出モデル (InceptionResnetV1)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def extract_face_embedding(self, image_path):
        """
        画像から顔の特徴ベクトル(embedding)を抽出
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            numpy array: 512次元の特徴ベクトル、または None (顔が検出できない場合)
        """
        try:
            img = Image.open(image_path).convert('RGB')
            
            # 顔を検出してクロップ
            img_cropped = self.mtcnn(img)
            
            if img_cropped is None:
                print(f"顔が検出できませんでした: {image_path}")
                return None
            
            # 特徴ベクトルを抽出
            img_cropped = img_cropped.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.resnet(img_cropped)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"エラー ({image_path}): {e}")
            return None
    
    def extract_features_from_directory(self, image_dir, output_file):
        """
        ディレクトリ内の全画像から特徴量を抽出
        
        Args:
            image_dir: 画像ディレクトリのパス
            output_file: 出力するJSONファイルのパス
        """
        image_dir = Path(image_dir)
        output_file = Path(output_file)
        
        if not image_dir.exists():
            print(f"ディレクトリが見つかりません: {image_dir}")
            return
        
        # 画像ファイルを取得
        extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in extensions]
        
        print(f"{len(image_files)}枚の画像から特徴量を抽出します...")
        
        features_dict = {}
        
        for i, img_path in enumerate(image_files, 1):
            print(f"処理中 ({i}/{len(image_files)}): {img_path.name}")
            
            embedding = self.extract_face_embedding(img_path)
            
            if embedding is not None:
                features_dict[img_path.name] = embedding.tolist()
        
        # 結果を保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(features_dict, f)
        
        print(f"\n完了: {len(features_dict)}枚の特徴量を {output_file} に保存しました")
        print(f"失敗: {len(image_files) - len(features_dict)}枚")

if __name__ == "__main__":
    extractor = FaceFeatureExtractor()
    extractor.extract_features_from_directory(
        image_dir="data/faces",
        output_file="data/features.json"
    )
