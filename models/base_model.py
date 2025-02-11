import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """모든 HAI 모델의 기본 클래스"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def predict(self, x):
        """예측 실행"""
        self.eval()
        with torch.no_grad():
            return self.forward(x) 