import torch
from torch.utils.data import DataLoader
import sys, io
import argparse
from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from config.config_hai import Config, ModelConfig
from utils.data_loader import load_dataset, HAIDataset
from utils.preprocessor import HAIPreprocessor
from models.gru_model import StackedGRU
from utils.feature_selector import FeatureSelector
import dateutil.parser
import numpy as np
from TaPR_pkg import etapr
from datetime import datetime
import pandas as pd
from torch.utils.data import random_split
from models.transformer_model import TimeSeriesTransformer
from models.lstm_model import StackedLSTM  # import 추가

def parse_args():
    parser = argparse.ArgumentParser(description='HAI 모델 학습')
    
    # 모델 파라미터
    parser.add_argument('--window-size', type=int, default=50,
                        help='윈도우 크기 (기본값: 90)')
    parser.add_argument('--window-given', type=int, default=49,
                        help='입력으로 사용할 윈도우 크기 (기본값: 89)')
    parser.add_argument('--n-hiddens', type=int, default=100,
                        help='히든 레이어 크기 (기본값: 100)')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='레이어 수 (기본값: 3)')
    
    # 학습 파라미터
    parser.add_argument('--batch-size', type=int, default=64,
                        help='배치 크기 (기본값: 512)')
    parser.add_argument('--n-epochs', type=int, default=3,
                        help='에폭 수 (기본값: 3)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='학습률 (기본값: 0.001)')
    
    # 평가 파라미터
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='이상 탐지 임계값 (기본값: 0.1)')
    
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0

def train(dataset, model, config, model_dir, timestamp):
    # 데이터셋을 학습/검증 세트로 분할 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 모델 summary 생성 및 저장
    sample = dataset[0]
    n_features = sample['given'].shape[-1]
    input_size = (config.batch_size, config.window_given, n_features)
    model_summary = summary(model, input_size=input_size, device='cuda')
    
    # summary를 파일로 저장
    with open(model_dir / f'model_summary_{timestamp}.txt', 'w') as f:
        f.write("Model Configuration:\n")
        f.write(str(vars(config)))
        f.write("\n\nModel Summary:\n")
        old_stdout = sys.stdout
        summary_buffer = io.StringIO()
        sys.stdout = summary_buffer
        print(model_summary)
        sys.stdout = old_stdout
        f.write(summary_buffer.getvalue())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)  # L2 정규화 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_fn = torch.nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    epochs = trange(config.n_epochs, desc="training")
    train_losses = []
    val_losses = []
    
    for e in epochs:
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            guess = model(given)
            answer = batch["answer"].cuda()
            loss = loss_fn(answer, guess)
            loss.backward()
            # # Gradient Clipping 추가
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_loss += loss.item()
            optimizer.step()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                given = batch["given"].cuda()
                guess = model(given)
                answer = batch["answer"].cuda()
                loss = loss_fn(answer, guess)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        
        epochs.set_postfix_str(f"train_loss: {avg_train_loss:.6f}, val_loss: {avg_val_loss:.6f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 최적의 모델 상태로 복원
    model.load_state_dict(early_stopping.best_state)
    # best_epoch는 early_stopping이 발생했을 때의 에폭
    best_epoch = len(train_losses) - early_stopping.counter
   
    
    return {
        "state": early_stopping.best_state,
        "total_epochs": len(train_losses),    # 총 학습된 에폭 수
        "best_epoch": best_epoch,             # 최적의 성능을 보인 에폭
        "loss_history": train_losses,         # 학습 손실 기록
        "loss": early_stopping.best_loss,
        "val_losses": val_losses
    }                

def create_directories():
    # 모델 저장 디렉토리
    model_dir = Path("models/gru")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 저장 디렉토리
    result_dir = Path("results/gru")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir, result_dir

def create_model(model_type, n_features, model_config):
    """모델 생성 함수"""
    if model_type == "gru":
        return StackedGRU(
            n_features=n_features,
            n_hiddens=model_config.n_hiddens,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout
        ).cuda()
    elif model_type == "transformer":
        return TimeSeriesTransformer(
            n_features=n_features,
            d_model=model_config.d_model,
            nhead=model_config.nhead,
            num_layers=model_config.n_layers,
            dropout=model_config.dropout
        ).cuda()
    elif model_type == "lstm":
        return StackedLSTM(
            n_features=n_features,
            n_hiddens=model_config.n_hiddens,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            bidirectional=model_config.bidirectional
        ).cuda()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main(model_config=None, experiment_name=None, model_type="gru"):
    """
    Args:
        model_config: ModelConfig 객체 (None이면 command line args에서 생성)
        experiment_name: 실험 이름 (파일 저장시 사용)
        model_type: 모델 유형 (기본값: "gru")
    """
    if model_config is None:
        args = parse_args()
        model_config = ModelConfig(
            window_size=args.window_size,
            window_given=args.window_given,
            n_hiddens=args.n_hiddens,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            dropout=0.2
        )
    # 사용예시
    # python train.py --n-epochs 5 --batch-size 256 --learning-rate 0.0005    
    # 디렉토리 생성
    model_dir, result_dir = create_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 데이터 로드
    data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)
    
    # 전처리
    preprocessor = HAIPreprocessor()
    data['train'] = data['train'][data['train']['attack']!=1] # 정상데이터만 학습에 활용
    valid_columns = data['train'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
    preprocessor.fit(data['train'], valid_columns)
    
    train_processed = preprocessor.transform(data['train'], valid_columns) # 다같은 칼럼 제거, 정규화, EWM 적용 
    # Feature Selector 초기화
    same_col = [col for col in valid_columns if col not in train_processed.columns] # 8개
    n_features = train_processed.shape[-1]
    # ['P1_PCV02D', 'P2_Auto', 'P2_Emgy', 'P2_On', 'P2_TripEx', 'P3_LH', 'P3_LL', 'P4_HT_PS']
    # selector = FeatureSelector()
    # # 상관계수 계산
    # selector.calculate_correlation(train_processed)
    # selected_features = selector.select_features(
    # train_processed,
    # correlation_threshold=0.99,
    # mutual_info_threshold=None)
    # selector.save_results('feature_analysis_results', threshold=0.99)
    # train_processed_selected = train_processed[selected_features]
    # drop_features = [col for col in train_processed.columns if col not in selected_features] # 9개 (99%이상 선형상관도)
    # 데이터셋 생성
    train_dataset = HAIDataset(
        data['train'][Config.TIMESTAMP_FIELD],
        train_processed,
        window_size=model_config.window_size,
        window_given=model_config.window_given,
        stride=10 # 데이터가 너무 많아서 10칸씩 띄워서 사용
    )

    # 모델 생성 
    model = create_model(model_type, n_features, model_config)
    
    # 학습
    best_model = train(train_dataset, model, model_config, model_dir, timestamp)
    
    # 모델 저장
    model_path = model_dir / f"gru_model_{experiment_name}_{timestamp}.pt"
    torch.save({
        "state": best_model["state"],
        "best_epoch": best_model["best_epoch"],
        "loss_history": best_model["loss_history"],
        "config": model_config.__dict__
    }, model_path)
    # 학습 결과 저장
    with open(result_dir / f"train_results_gru_{experiment_name}_{timestamp}.txt", 'w') as f:
        f.write(f"Training Configuration:\n")
        for key, value in model_config.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest Epoch: {best_model['best_epoch']}")
        f.write(f"\nBest Loss: {best_model['loss']:.6f}")
     
    # 손실 그래프 저장
    plt.figure(figsize=(16, 4))
    plt.title("Training Loss Graph")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(best_model["loss_history"])
    plt.plot(best_model["val_losses"])
    plt.plot(best_model['best_epoch']-1, best_model['loss'], 'ro')
    plt.savefig(result_dir / f"loss_history_gru_{experiment_name}_{timestamp}.png")
    plt.close()
    
    return model_path, experiment_name

def find_optimal_threshold(anomaly_scores, true_labels, start=0.0, end=1.0, steps=100):
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
    
    return optimal_threshold, best_f1, best_metrics

# 라벨 생성
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def test(model_path="models/gru/gru_model.pt",experiment_name='', find_threshold=False):
    """학습된 모델을 불러와 테스트를 수행"""
    # 결과 저장 디렉토리
    _, result_dir = create_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 모델 설정 불러오기
    if isinstance(model_path, (str, Path)):
        saved_model = torch.load(model_path)
    else:
        # BytesIO나 다른 파일 객체인 경우
        model_path.seek(0)  # 파일 포인터를 처음으로 되돌림
        saved_model = torch.load(model_path)
        
    model_config = ModelConfig(**saved_model["config"])
    
    # 데이터 로드
    data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)
    
    # 전처리
    preprocessor = HAIPreprocessor()
    valid_columns = data['test'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
    data['train'] = data['train'][data['train']['attack']!=1] # 정상데이터만 학습에 활용
    preprocessor.fit(data['train'], valid_columns)  # train 데이터로 fit
    
    # train 데이터로 전처리했을 때의 특성 목록을 먼저 얻음
    train_processed = preprocessor.transform(data['train'], valid_columns)
    train_features = train_processed.columns.tolist()
    
    # test 데이터 전처리 시 train과 동일한 특성만 사용
    
    test_processed = preprocessor.transform(data['test'], valid_columns)  # train과 동일한 특성만 선택
    # 'P1_PCV02D'는 train에서 값이 다 같아서 학습안됨. test에서도 빼고 평가 필요
    test_processed = test_processed[train_features]
    n_features = test_processed.shape[-1]
    # 테스트 데이터셋 생성 (stride=1로 모든 데이터 포인트 검사)
    test_dataset = HAIDataset(
        data['test'][Config.TIMESTAMP_FIELD],
        test_processed,
        window_size=model_config.window_size,
        window_given=model_config.window_given,
        stride=1,
        attacks=data['test'][Config.ATTACK_FIELD]
    )
    
    # 모델 생성 및 가중치 로드
    model = create_model(model_type="gru", n_features=n_features, model_config=model_config)
    model.load_state_dict(saved_model["state"])
    model.eval()
    
    # 추론 수행
    def inference(dataset, model, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        ts, dist, att = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                given = batch["given"].cuda()
                answer = batch["answer"].cuda()
                guess = model(given)
                ts.append(np.array(batch["ts"]))
                dist.append(torch.abs(answer - guess).cpu().numpy())
                att.append(np.array(batch["attack"]))
        return (
            np.concatenate(ts),
            np.concatenate(dist),
            np.concatenate(att),
        )
    
    # 테스트 데이터에 대한 예측 수행
    check_ts, check_dist, check_att = inference(test_dataset, model, model_config.batch_size)
    
    # 이상 점수 계산
    anomaly_score = np.mean(check_dist, axis=1)
    
    if find_threshold:
        # 최적의 threshold 찾기
        optimal_threshold, best_f1, best_metrics = find_optimal_threshold(
            anomaly_scores=anomaly_score,
            true_labels=check_att,
            start=0.0,
            end=max(anomaly_score),  # 최대 이상 점수를 기준으로 설정
            steps=100
        )
        
        # 최적의 threshold로 라벨 생성
        labels = put_labels(anomaly_score, optimal_threshold)
    else:
        # 기존 설정된 threshold 사용
        labels = put_labels(anomaly_score, model_config.threshold)
        optimal_threshold = model_config.threshold    
    # 그래프 생성 및 저장
    def check_graph(xs, att, result_dir, timestamp, piece=2):
        l = xs.shape[0]
        chunk = l // piece
        fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
        for i in range(piece):
            L = i * chunk
            R = min(L + chunk, l)
            xticks = range(L, R)
            axs[i].plot(xticks, xs[L:R], label='Anomaly Score')
            if len(xs[L:R]) > 0:
                peak = max(xs[L:R])
                axs[i].plot(xticks, att[L:R] * peak * 0.3, label='Attack', alpha=0.5)
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()
        plt.savefig(result_dir / f"test_anomaly_score_gru_{experiment_name}_{timestamp}.png")
        plt.close()

    check_graph(anomaly_score, check_att, result_dir, timestamp, piece=3)

    # attack_labels = put_labels(check_att, threshold=optimal_threshold)
       # 윈도우 방식의 단점
    # 탐지 모델이 윈도우 방식으로 판단을 진행했기 때문에,
    # 1. 첫 시작의 몇 초는 판단을 내릴 수 없고
    # 2. 데이터셋 중간에 시간이연속되지 않는 구간에 대해서는 판단을 내릴 수 없습니다.
    # 빈 곳은 정상(0) 표기하고 나머지는 모델의 판단(정상 0, 비정상 1)을 채워줍니다.
    # 빈 구간 채우기
    def fill_blank(check_ts, labels, total_ts):
        def ts_generator():
            for t in total_ts:
                yield dateutil.parser.parse(t)

        def label_generator():
            for t, label in zip(check_ts, labels):
                yield dateutil.parser.parse(t), label

        g_ts = ts_generator()
        g_label = label_generator()
        final_labels = []

        try:
            current = next(g_ts)
            ts_label, label = next(g_label)
            while True:
                if current > ts_label:
                    ts_label, label = next(g_label)
                    continue
                elif current < ts_label:
                    final_labels.append(0)
                    current = next(g_ts)
                    continue
                final_labels.append(label)
                current = next(g_ts)
                ts_label, label = next(g_label)
        except StopIteration:
            return np.array(final_labels, dtype=np.int8)
    
    # 최종 예측 라벨 생성
    final_labels = fill_blank(check_ts, labels, np.array(data['test'][Config.TIMESTAMP_FIELD]))
    final_attack_labels = np.array(data['test'][Config.ATTACK_FIELD])

    # 최종 TaPR 평가 수행
    tapr = etapr.evaluate(anomalies=final_attack_labels, predictions=final_labels)

    # 상세 결과를 CSV로 저장
    results_df = pd.DataFrame({
        'timestamp': check_ts,
        'anomaly_score': anomaly_score,
        'predicted_label': labels,
        'ground_truth': check_att
    })
    results_df.to_csv(result_dir / f"detailed_results_gru_{experiment_name}_{timestamp}.csv", index=False)
    
    # 메트릭스를 CSV로 저장
    metrics_df = pd.DataFrame({
        'metric': ['F1', 'TaP', 'TaR', 'Detected_Anomalies'],
        'value': [tapr['f1'], tapr['TaP'], tapr['TaR'], len(tapr['Detected_Anomalies'])]
    })
    metrics_df.to_csv(result_dir / f"metrics_gru_{experiment_name}_{timestamp}.csv", index=False)
    
    # 결과 출력 및 저장
     # 결과 파일에 threshold 정보 추가
    result_file = result_dir / f"test_results_gru_{experiment_name}_{timestamp}.txt"
    with open(result_file, 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Threshold: {optimal_threshold:.6f}\n")
        f.write(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})\n")
        f.write(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}\n")
        f.write(f"\nDetected anomalies: {tapr['Detected_Anomalies']}")
    
    print("\nTest Results:")
    print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")
    print(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}")
    print(f"Results saved in: {result_dir}")
    
    return {
        'anomaly_score': anomaly_score,
        'predictions': final_labels,
        'ground_truth': final_attack_labels,
        'metrics': tapr,
        'threshold': optimal_threshold
    }

if __name__ == "__main__":
    # experiment_name = "baseline_b256_w10"
    # model_path, experiment_name = main(experiment_name=experiment_name)# 인자 없이 호출하면 args에서 설정을 가져옴
    model_path = str(project_root) + "/models/gru/gru_model_batch128_h200_w10_20250212_112246.pt"
    experiment_name = "batch128_h200_w10_20250212_112246"
    test_results = test(model_path, experiment_name, find_threshold=True)  