from pathlib import Path
from dataclasses import dataclass
from typing import Optional

class Config:
    # 데이터 관련 (변경되지 않는 기본 설정)
    DATA_DIR = "//147.47.239.145/SHRM-Motor/[2] 데이터 및 코드/[2] 오픈 데이터셋/[5] 기타/[4] HAI dataset (Anomaly Detection)/data/"#"C:/Users/pchee/GitRepo/Dataset/[4] HAI Dataset/data/"
    HAI_VERSION = "HAI 1.0/"
    TIMESTAMP_FIELD = "time"
    ATTACK_FIELD = "attack"
    USELESS_FIELDS = ["attack_P1", "attack_P2", "attack_P3"]

@dataclass
class BaseModelConfig:
    """모든 모델의 공통 설정"""
    window_size: int = 60
    window_given: int = 59
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 0.001
    dropout: float = 0.1
    threshold: float = 0.1

@dataclass
class RNNModelConfig(BaseModelConfig):
    """GRU와 LSTM 공통 설정"""
    n_hiddens: int = 100
    n_layers: int = 3
    bidirectional: bool = False  # LSTM에서만 사용됨

@dataclass
class TransformerModelConfig(BaseModelConfig):
    """Transformer 전용 설정"""
    d_model: int = 128
    nhead: int = 8
    n_layers: int = 3
    
def create_model_config(model_type: str, **kwargs) -> BaseModelConfig:
    """모델 타입에 따른 설정 객체 생성"""
    if model_type in ["gru", "lstm"]:
        config = RNNModelConfig(**kwargs)
        if model_type == "lstm":
            config.bidirectional = kwargs.get('bidirectional', True)
        return config
    elif model_type == "transformer":
        return TransformerModelConfig(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# 기존 ModelConfig는 하위 호환성을 위해 유지
class ModelConfig(RNNModelConfig):
    """기존 ModelConfig (하위 호환성 유지)"""
    pass

@dataclass
class XGBoostConfig:
    # 데이터 관련 설정
    window_size: int = 10
    window_given: int = 9
    
    # XGBoost 모델 파라미터
    n_models: int = 3 # 앙상블 모델 개수
    learning_rate: float = 0.001
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    
    # GPU 설정
    use_gpu: bool = True
    gpu_id: int = 0

@dataclass
class TransformerConfig:
    # 데이터 관련 설정
    window_size: int = 60
    window_given: int = 59
    
    # Transformer 모델 파라미터
    d_model: int = 128
    nhead: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    
    # 학습 관련
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 0.0001
    
    # 평가 관련
    threshold: float = 0.1

@dataclass
class LSTMConfig:
    # 데이터 관련 설정
    window_size: int = 60
    window_given: int = 59
    
    # LSTM 모델 파라미터
    n_hiddens: int = 200
    n_layers: int = 3
    dropout: float = 0.1
    bidirectional: bool = True
    
    # 학습 관련
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 0.0001
    
    # 평가 관련
    threshold: float = 0.1