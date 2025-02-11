import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

class FeatureSelector:
    def __init__(self):
        self.correlation_matrix = None
        self.mutual_info_scores = None
        self.selected_features = None
        
    def calculate_correlation(self, df: pd.DataFrame, method='pearson'):
        """
        피어슨 또는 스피어만 상관계수를 계산합니다.
        
        Args:
            df: 입력 데이터프레임
            method: 'pearson' 또는 'spearman'
        """
        self.correlation_matrix = df.corr(method=method)
        return self.correlation_matrix
    
    def calculate_mutual_info(self, df: pd.DataFrame, target_col: str):
        """
        각 feature와 target 변수 간의 mutual information을 계산합니다.
        
        Args:
            df: 입력 데이터프레임
            target_col: target 변수명
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.mutual_info_scores = pd.Series(
            mutual_info_regression(X, y),
            index=X.columns
        )
        return self.mutual_info_scores
    
    def select_features(self, df: pd.DataFrame, correlation_threshold=0.8, 
                       mutual_info_threshold=0.1):
        """
        상관계수와 mutual information을 기반으로 feature를 선택합니다.
        
        Args:
            df: 입력 데이터프레임
            correlation_threshold: 상관계수 임계값 (이 값 이상인 경우 한 변수만 선택)
            mutual_info_threshold: mutual information 임계값 (이 값 이상인 변수만 선택)
        """
        # 높은 상관관계를 가진 변수들 중 더 중요한 변수 선택
        corr_matrix = self.correlation_matrix.abs()
        
        # 변수별 중요도 점수 계산 (mutual info가 있으면 사용, 없으면 분산 사용)
        if self.mutual_info_scores is not None:
            importance_scores = self.mutual_info_scores
        else:
            importance_scores = df.var()
        
        features_to_drop = set()
        
        # 상관계수가 높은 변수쌍들을 찾아서 처리
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i,j] > correlation_threshold:
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    # 중요도가 낮은 변수를 제거 대상으로 선정
                    if importance_scores[feat1] > importance_scores[feat2]:
                        features_to_drop.add(feat2)
                    else:
                        features_to_drop.add(feat1)
        # Mutual Information이 높은 변수들 선택 MI 점수가 없는 경우: 분산값 사용 (분산이 큰 변수가 더 많은 정보를 담고 있다고 가정)
        # Mutual Information 조건 적용
        if self.mutual_info_scores is not None:
            selected_by_mi = set(self.mutual_info_scores[
                self.mutual_info_scores > mutual_info_threshold
            ].index)
            
            # 최종 feature 선택: 제거 대상이 아니면서 MI 조건을 만족하는 변수들
            self.selected_features = [
                f for f in df.columns 
                if f not in features_to_drop and f in selected_by_mi
            ]
        else:
            # MI 점수가 없는 경우는 상관관계만 고려
            self.selected_features = [
                f for f in df.columns 
                if f not in features_to_drop
            ]
            
        return self.selected_features
    
    def plot_correlation_heatmap(self, save_path=None, threshold=0.8):
        """
        상관계수 히트맵을 그립니다.
        
        Args:
            save_path: 결과를 저장할 경로 (선택사항)
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        plt.title('Feature Correlation Heatmap_threshold='+str(threshold))
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_mutual_info(self, save_path=None, threshold=0.1):
        """
        Mutual Information 막대 그래프를 그립니다.
        
        Args:
            save_path: 결과를 저장할 경로 (선택사항)
        """
        if self.mutual_info_scores is None:
            raise ValueError("Mutual information scores haven't been calculated yet.")
            
        plt.figure(figsize=(12, 6))
        self.mutual_info_scores.sort_values(ascending=True).plot(kind='barh')
        plt.title('Mutual Information Scores_threshold='+str(threshold))
        plt.xlabel('Mutual Information')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def save_results(self, output_dir: str, threshold=0.8):
        """
        분석 결과를 저장합니다.
        
        Args:
            output_dir: 결과를 저장할 디렉토리 경로
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 상관계수 행렬 저장
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(output_path / 'correlation_matrix.csv')
            self.plot_correlation_heatmap(output_path / 'correlation_heatmap.png', threshold=threshold)
            
        # Mutual Information 점수 저장
        if self.mutual_info_scores is not None:
            self.mutual_info_scores.to_csv(output_path / 'mutual_info_scores.csv')
            self.plot_mutual_info(output_path / 'mutual_info_plot.png', threshold=threshold)
            
        # 선택된 feature 목록 저장
        if self.selected_features is not None:
            filename = f'selected_features_th{threshold}.txt'
            with open(output_path.joinpath(filename), 'w') as f:
                f.write('\n'.join(self.selected_features))