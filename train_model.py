"""
好み予測モデルを訓練するスクリプト
評価データと特徴量から機械学習モデルを訓練
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class PreferenceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self, features_file, ratings_file):
        """
        特徴量と評価データを読み込む
        
        Args:
            features_file: 特徴量JSONファイル
            ratings_file: 評価データJSONファイル
            
        Returns:
            X: 特徴量配列
            y: 評価スコア配列
        """
        # 特徴量を読み込み
        with open(features_file, 'r') as f:
            features_dict = json.load(f)
        
        # 評価データを読み込み
        with open(ratings_file, 'r', encoding='utf-8') as f:
            ratings_dict = json.load(f)
        
        # 両方に存在するデータのみを使用
        common_files = set(features_dict.keys()) & set(ratings_dict.keys())
        
        if not common_files:
            raise ValueError("特徴量と評価データに共通する画像がありません")
        
        print(f"学習データ数: {len(common_files)}枚")
        
        X = []
        y = []
        filenames = []
        
        for filename in common_files:
            X.append(features_dict[filename])
            y.append(ratings_dict[filename])
            filenames.append(filename)
        
        return np.array(X), np.array(y), filenames
    
    def train(self, X, y, model_type='random_forest'):
        """
        モデルを訓練
        
        Args:
            X: 特徴量
            y: 評価スコア
            model_type: モデルの種類 ('random_forest' or 'svr')
        """
        if len(X) < 10:
            print("警告: 学習データが少なすぎます（最低10枚推奨）")
        
        # データを正規化
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練データとテストデータに分割
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            # データが少ない場合は全データで訓練
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
            print("データ数が少ないため、全データで訓練します")
        
        # モデルを選択
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError(f"未知のモデルタイプ: {model_type}")
        
        # 訓練
        print(f"モデルを訓練中 ({model_type})...")
        self.model.fit(X_train, y_train)
        
        # 評価
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n=== 訓練結果 ===")
        print(f"訓練データ RMSE: {train_rmse:.3f}")
        print(f"テストデータ RMSE: {test_rmse:.3f}")
        print(f"訓練データ R²: {train_r2:.3f}")
        print(f"テストデータ R²: {test_r2:.3f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
    
    def save_model(self, filepath):
        """モデルを保存"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        
        print(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath):
        """モデルを読み込み"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        
        print(f"モデルを読み込みました: {filepath}")
    
    def predict(self, X):
        """予測"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def plot_predictions(y_true, y_pred, save_path='data/prediction_plot.png'):
    """予測結果をプロット"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUIなしバックエンド
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([1, 10], [1, 10], 'r--', lw=2)
        plt.xlabel('実際の評価', fontsize=12)
        plt.ylabel('予測評価', fontsize=12)
        plt.title('予測精度', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 11)
        plt.ylim(0, 11)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"予測結果のグラフを保存しました: {save_path}")
        plt.close()
    except Exception as e:
        print(f"グラフ作成をスキップしました: {e}")

if __name__ == "__main__":
    model = PreferenceModel()
    
    # データを読み込み
    X, y, filenames = model.load_data(
        features_file="data/features.json",
        ratings_file="data/ratings.json"
    )
    
    # モデルを訓練
    results = model.train(X, y, model_type='random_forest')
    
    # 予測結果をプロット
    plot_predictions(results['y_test'], results['y_pred_test'])
    
    # モデルを保存
    model.save_model("data/preference_model.pkl")
