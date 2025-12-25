"""
顔画像評価ツール
顔画像を表示してユーザーに評価してもらい、データを保存する
"""

import os
import json
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class FaceEvaluator:
    def __init__(self, image_dir="data/faces", output_file="data/ratings.json"):
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.ratings = self.load_ratings()
        self.image_files = self.get_image_files()
        self.current_index = 0
        
        self.setup_gui()
        
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
            self.image_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        extensions = ['.jpg', '.jpeg', '.png']
        files = [f for f in self.image_dir.iterdir() 
                if f.suffix.lower() in extensions]
        return sorted(files)
    
    def setup_gui(self):
        """GUIのセットアップ"""
        self.root = tk.Tk()
        self.root.title("顔の好み評価システム")
        self.root.geometry("800x900")
        
        # 画像表示エリア
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)
        
        # 進捗表示
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.progress_label.pack()
        
        # ファイル名表示
        self.filename_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.filename_label.pack()
        
        # 評価スライダー
        self.rating_frame = tk.Frame(self.root)
        self.rating_frame.pack(pady=20)
        
        tk.Label(self.rating_frame, text="評価 (1-10):", font=("Arial", 14)).pack()
        self.rating_var = tk.IntVar(value=5)
        self.rating_slider = tk.Scale(
            self.rating_frame, 
            from_=1, 
            to=10, 
            orient=tk.HORIZONTAL,
            variable=self.rating_var,
            length=400,
            font=("Arial", 12)
        )
        self.rating_slider.pack()
        
        # ボタンエリア
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        tk.Button(
            button_frame, 
            text="← 前へ", 
            command=self.prev_image,
            font=("Arial", 12),
            width=10
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame, 
            text="評価して次へ →", 
            command=self.save_and_next,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame, 
            text="スキップ", 
            command=self.next_image,
            font=("Arial", 12),
            width=10
        ).pack(side=tk.LEFT, padx=10)
        
        # 統計情報
        self.stats_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.stats_label.pack(pady=10)
        
    def load_and_display_image(self):
        """現在の画像を読み込んで表示"""
        if not self.image_files:
            self.image_label.config(
                text="画像がありません\ndata/faces/ フォルダに画像を配置してください",
                font=("Arial", 14)
            )
            return
        
        if self.current_index >= len(self.image_files):
            self.show_completion()
            return
        
        img_path = self.image_files[self.current_index]
        
        # 画像を読み込んで表示
        img = Image.open(img_path)
        img.thumbnail((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
        
        # 進捗とファイル名を更新
        self.progress_label.config(
            text=f"{self.current_index + 1} / {len(self.image_files)}"
        )
        self.filename_label.config(text=img_path.name)
        
        # 既存の評価があれば読み込む
        if img_path.name in self.ratings:
            self.rating_var.set(self.ratings[img_path.name])
        else:
            self.rating_var.set(5)
        
        self.update_stats()
    
    def save_and_next(self):
        """評価を保存して次へ"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        img_path = self.image_files[self.current_index]
        self.ratings[img_path.name] = self.rating_var.get()
        self.save_ratings()
        
        self.next_image()
    
    def next_image(self):
        """次の画像へ"""
        self.current_index += 1
        self.load_and_display_image()
    
    def prev_image(self):
        """前の画像へ"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_and_display_image()
    
    def update_stats(self):
        """統計情報を更新"""
        if not self.ratings:
            self.stats_label.config(text="評価数: 0")
            return
        
        ratings_list = list(self.ratings.values())
        avg = sum(ratings_list) / len(ratings_list)
        self.stats_label.config(
            text=f"評価済み: {len(self.ratings)}枚 | 平均点: {avg:.1f}"
        )
    
    def show_completion(self):
        """完了画面を表示"""
        self.image_label.config(
            text=f"全ての画像の評価が完了しました！\n\n評価数: {len(self.ratings)}枚",
            font=("Arial", 16)
        )
        self.progress_label.config(text="")
        self.filename_label.config(text="")
    
    def run(self):
        """アプリケーションを起動"""
        self.load_and_display_image()
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceEvaluator()
    app.run()
