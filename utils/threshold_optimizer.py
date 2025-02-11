import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from TaPR_pkg import etapr

class ThresholdOptimizer:
    def __init__(self):
        self.best_threshold = None
        self.best_f1 = None
        self.best_metrics = None
        self.thresholds = None
        
    def find_optimal_threshold(self, anomaly_scores, true_labels, start=0.0, end=1.0, steps=100):
        """
        F1 스코어를 최대화하는 최적의 threshold를 찾습니다.
        
        Args:
            anomaly_scores: 이상 점수 배열
            true_labels: 실제 라벨 배열
            start: 탐색할 threshold의 시작값
            end: 탐색할 threshold의 끝값
            steps: 탐색할 threshold의 개수
        
        Returns:
            optimal_threshold: 최적의 threshold 값
            best_f1: 최적의 F1 스코어
            best_metrics: 최적의 threshold에서의 메트릭스 (TaP, TaR 등)
        """
        thresholds = np.linspace(start, end, steps)
        best_f1 = 0
        optimal_threshold = 0
        best_metrics = None
    
        print("\nFinding optimal threshold...")
        for threshold in tqdm(thresholds, desc="Threshold search"):
            # 예측 라벨 생성
            pred_labels = put_labels(anomaly_scores, threshold)
            
            # TaPR 평가
            metrics = etapr.evaluate(anomalies=true_labels, predictions=pred_labels)
            f1 = metrics['f1']
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
                best_metrics = metrics
        
        print(f"\nOptimal threshold found: {optimal_threshold:.4f}")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"TaP: {best_metrics['TaP']:.4f}, TaR: {best_metrics['TaR']:.4f}")
        self.best_threshold = optimal_threshold
        self.best_f1 = best_f1
        self.best_metrics = best_metrics
        return optimal_threshold, best_f1, best_metrics
# 라벨 생성
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs