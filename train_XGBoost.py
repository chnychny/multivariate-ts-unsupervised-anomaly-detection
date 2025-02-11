import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib
import argparse
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from datetime import datetime
import pandas as pd
from config.config_hai import Config, XGBoostConfig
from utils.data_loader import load_dataset
from utils.preprocessor import HAIPreprocessor
from models.xgb_model import MultiOutputXGBoost
from TaPR_pkg import etapr
import dateutil.parser
from utils.threshold_optimizer import ThresholdOptimizer

def create_directories():
    # 모델 저장 디렉토리
    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 저장 디렉토리
    result_dir = Path("results/xgboost")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir, result_dir
def parse_args():
    parser = argparse.ArgumentParser(description='HAI XGBoost 모델 학습')
    
    # 모델 파라미터
    parser.add_argument('--window-size', type=int, default=10,
                        help='윈도우 크기 (기본값: 50)')
    parser.add_argument('--window-given', type=int, default=9,
                        help='입력으로 사용할 윈도우 크기 (기본값: 49)')
    
    # XGBoost 파라미터
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='학습률 (기본값: 0.1)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='트리 최대 깊이 (기본값: 6)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='트리 개수 (기본값: 100)')
    parser.add_argument('--n-models', type=int, default=3,
                        help='앙상블 모델 개수 (기본값: 3)')
    # 평가 파라미터
    # parser.add_argument('--threshold', type=float, default=0.1,
    #                     help='이상 탐지 임계값 (기본값: 0.1)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='사용할 GPU 장치 번호 (기본값: 0)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='GPU 사용 여부')    
    return parser.parse_args()

def prepare_data_for_xgb(dataset, window_size, window_given):
    """
    DataFrame을 시계열 윈도우로 변환하는 함수
    
    Args:
        dataset: 입력 DataFrame (shape: [n_samples, n_features])
        window_size: 전체 윈도우 크기
        window_given: 입력으로 사용할 윈도우 크기
    """
    X_list, y_list = [], []
    
    # 전체 데이터에서 윈도우 크기만큼 슬라이딩하며 데이터 생성
    for i in range(len(dataset) - window_size + 1):
        # 입력 윈도우 (given)
        X = dataset.iloc[i:i + window_given].values.flatten()  # window_given 크기의 윈도우를 1차원으로
        
        # 정답 윈도우 (answer)
        y = dataset.iloc[i + window_given:i + window_size].values.flatten()  # 나머지 윈도우를 1차원으로
        
        X_list.append(X)
        y_list.append(y)
    
    return np.array(X_list), np.array(y_list)

def train(train_dataset, model_config):
    # 데이터 준비
    X_train, y_train = prepare_data_for_xgb(train_dataset, model_config.window_size, model_config.window_given)
    # GPU 관련 파라미터 설정
    xgb_params = {
        'n_features': y_train.shape[1],
        'learning_rate': model_config.learning_rate,
        'max_depth': model_config.max_depth,
        'n_estimators': model_config.n_estimators,
    }
        
    if model_config.use_gpu:
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': model_config.gpu_id,
            'predictor': 'gpu_predictor'
        })
    
    # 모델 생성
    model = MultiOutputXGBoost(
        n_features=y_train.shape[1],
        n_models=model_config.n_models,
        learning_rate=model_config.learning_rate,
        max_depth=model_config.max_depth,
        n_estimators=model_config.n_estimators
    )
    
    # 학습
    model.fit(X_train, y_train)
    
    # 학습 결과 평가
    train_pred = model.predict(X_train)
    train_loss = mean_squared_error(y_train, train_pred)
    
    return model, train_loss

def main():
    args = parse_args()
    model_config = XGBoostConfig(
        window_size=args.window_size,
        window_given=args.window_given,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        # threshold=args.threshold,
        n_models=args.n_models,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id
    )
    # 디렉토리 생성
    model_dir, result_dir = create_directories()
    
    # 현재 시간을 파일명에 사용
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 데이터 로드 및 전처리
    data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)
    preprocessor = HAIPreprocessor()
    data['train'] = data['train'][data['train']['attack']!=1] # 정상데이터만 학습에 활용
    valid_columns = data['train'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
    preprocessor.fit(data['train'], valid_columns)
    train_processed = preprocessor.transform(data['train'], valid_columns)
    
    # 모델 학습
    model, train_loss = train(train_processed, model_config)
    
    # 모델 저장
    model_path = model_dir / f"xgb_model_w10_{timestamp}.joblib"
    joblib.dump({
        "model": model,
        "config": model_config.__dict__,
        "train_loss": train_loss
    }, model_path)
    
    # 학습 결과 저장
    with open(result_dir / f"train_results_xgb_w10_{timestamp}.txt", 'w') as f:
        f.write(f"Training Configuration:\n")
        for key, value in model_config.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTraining Loss: {train_loss:.6f}")
    
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Model saved to: {model_path}")
    return model_path

def test(model_path="xgb_model.joblib"):
    # 결과 저장 디렉토리
    _, result_dir = create_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # 저장된 모델 로드
    saved = joblib.load(model_path)
    model = saved["model"]
    model_config = XGBoostConfig(**saved["config"])
    
    # 테스트 데이터 준비
    data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)
    preprocessor = HAIPreprocessor()
    data['train'] = data['train'][data['train']['attack']!=1] # 정상데이터만 학습에 활용
    valid_columns = data['test'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
    preprocessor.fit(data['train'], valid_columns)
    test_processed = preprocessor.transform(data['test'], valid_columns)
    
    # 예측 수행
    X_test, y_test = prepare_data_for_xgb(test_processed, model_config.window_size, model_config.window_given)
    predictions = model.predict(X_test)
    
    # 이상 점수 계산
    anomaly_score = np.mean(np.abs(y_test - predictions), axis=1)
    
    # 원본 데이터와 길이를 맞추기 위해 앞부분에 패딩 추가
    padding_length = model_config.window_size - 1
    padded_score = np.pad(anomaly_score, (padding_length, 0), mode='constant', constant_values=np.nan)
    
    # 원본 attack 라벨
    attack_labels = (data['test'][Config.ATTACK_FIELD] > 0.5).astype(int)
    
    # 최적 threshold 탐색 (패딩된 부분 제외)
    valid_indices = ~np.isnan(padded_score)
    optimizer = ThresholdOptimizer()
    optimal_threshold, best_f1, best_metrics = optimizer.find_optimal_threshold(
        anomaly_scores=padded_score[valid_indices],
        true_labels=attack_labels[valid_indices],
        start=0.0, end=1.0, steps=100
    )
    # 최적 threshold로 라벨 생성
    padded_labels = np.full_like(padded_score, -1, dtype=int)  # 패딩 부분은 -1로 초기화
    padded_labels[valid_indices] = (padded_score[valid_indices] > optimal_threshold).astype(int)
    
    # TaPR 평가 (패딩된 부분 제외)
    evaluation_indices = valid_indices & (padded_labels != -1)
    tapr = etapr.evaluate(
        anomalies=attack_labels[evaluation_indices], 
        predictions=padded_labels[evaluation_indices]
    )

    # DataFrame 생성 및 저장
    results_df = pd.DataFrame({
        'timestamp': data['test'][Config.TIMESTAMP_FIELD],
        'anomaly_score': padded_score,
        'predicted_label': padded_labels,
        'ground_truth': attack_labels
    })
    results_df.to_csv(result_dir / f"detailed_results_w10_{timestamp}.csv", index=False)
    
    # 결과 출력 및 저장
    print("\nTest Results:")
    print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")
    print(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}")
    
    result_file = result_dir / f"test_results_xgb_w10_{timestamp}.txt"
    with open(result_file, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.3f}\n")
        f.write(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})\n")
        f.write(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}\n")
    
    return {
        'anomaly_score': padded_score,
        'predictions': padded_labels,
        'ground_truth': attack_labels,
        'metrics': tapr,
        'optimal_threshold': optimal_threshold
    }

if __name__ == "__main__":
    model_path=main()
    # model_path = "models/xgboost/xgb_model_w20_20250203_183028.joblib"
    test_results = test(model_path)