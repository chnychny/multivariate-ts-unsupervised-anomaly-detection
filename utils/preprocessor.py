import pandas as pd
import numpy as np

class HAIPreprocessor:
    def __init__(self):
        self.tag_min = None
        self.tag_max = None
        self.selected_features = None  # 선택된 특성 목록 저장
        
    def fit(self, df, valid_columns):
        """학습 데이터로부터 정규화 파라미터 계산"""
        df = df[valid_columns]
        no_zero_columns = [col for col in df.columns if df[col].nunique() > 1]
        self.selected_features = no_zero_columns  # 선택된 특성 저장
        self.tag_min = df[no_zero_columns].min()
        self.tag_max = df[no_zero_columns].max()
        
    def normalize(self, df):
        """데이터 정규화"""
        if self.tag_min is None or self.tag_max is None:
            raise ValueError("정규화 전에 fit()을 먼저 실행하세요")
            
        ndf = df.copy()
        for c in df.columns:
            if c == "time":
                continue
            else:
                if self.tag_min[c] == self.tag_max[c]: # 같으면 전부 0으로 처리
                    ndf[c] = df[c] - self.tag_min[c]
                else: # 다르면 정규화 min, max
                    ndf[c] = (df[c] - self.tag_min[c]) / (self.tag_max[c] - self.tag_min[c])
        return ndf
    
    def transform(self, df, valid_columns, alpha=0.9):
        """정규화 및 EWM 적용"""
        if self.selected_features is None:
            raise ValueError("fit()을 먼저 실행하세요")
            
        df = df[self.selected_features] # 
        no_zero_columns = [col for col in df.columns if df[col].nunique() > 1]
        normalized = self.normalize(df[no_zero_columns])
        return normalized.ewm(alpha=alpha).mean() 
    
    def transform_with_time(self, df, valid_columns, alpha=0.9):
        """time 컬럼은 그대로 유지하고 나머지 변수에만 정규화 및 EWM 적용"""
        # time 컬럼 분리
        time_data = df['time']
        
        # time을 제외한 나머지 컬럼들에 대해 처리
        other_columns = [col for col in valid_columns if col != 'time']
        df_without_time = df[other_columns]
        
        # 전부 0인 column 제거
        no_zero_columns = [col for col in df_without_time.columns if df_without_time[col].nunique() > 1]
        
        # 정규화 및 EWM 적용
        processed_data = self.normalize(df_without_time[no_zero_columns])
        processed_data = processed_data.ewm(alpha=alpha).mean()
        
        # time 컬럼 다시 추가
        processed_data['time'] = time_data
        
        return processed_data