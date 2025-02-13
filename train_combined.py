import torch
from torch.utils.data import DataLoader, random_split
import sys, io
from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import dateutil.parser
from TaPR_pkg import etapr

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config_hai import Config, ModelConfig, create_model_config
from utils.data_loader import load_dataset, HAIDataset
from utils.preprocessor import HAIPreprocessor
from models.gru_model import StackedGRU
from models.transformer_model import TimeSeriesTransformer
from models.lstm_model import StackedLSTM
from utils.threshold_optimizer import ThresholdOptimizer

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

class ModelTrainer:
    def __init__(self, model_config, model_type="gru"):
        self.model_config = model_config
        self.model_type = model_type
        self.model_dir, self.result_dir = self._create_directories()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threshold_optimizer = ThresholdOptimizer()
        
    def _create_directories(self):
        model_dir = Path(f"models/{self.model_type}")
        result_dir = Path(f"results/{self.model_type}")
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        return model_dir, result_dir
    
    def create_model(self, n_features):
        if self.model_type == "gru":
            return StackedGRU(
                n_features=n_features,
                n_hiddens=self.model_config.n_hiddens,
                n_layers=self.model_config.n_layers,
                dropout=self.model_config.dropout
            ).cuda()
        elif self.model_type == "transformer":
            return TimeSeriesTransformer(
                n_features=n_features,
                d_model=self.model_config.d_model,
                nhead=self.model_config.nhead,
                num_layers=self.model_config.n_layers,
                dropout=self.model_config.dropout
            ).cuda()
        elif self.model_type == "lstm":
            return StackedLSTM(
                n_features=n_features,
                n_hiddens=self.model_config.n_hiddens,
                n_layers=self.model_config.n_layers,
                dropout=self.model_config.dropout,
                bidirectional=True
            ).cuda()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, train_dataset, model):
        # 데이터셋 분할
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=self.model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.model_config.batch_size)
        
        # 모델 summary 저장
        self._save_model_summary(model, train_dataset)
        
        # 학습 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.model_config.learning_rate, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        loss_fn = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=20, verbose=True)
        
        # 학습 루프
        epochs = trange(self.model_config.n_epochs, desc="Training")
        train_losses = []
        val_losses = []
        
        for epoch in epochs:
            # Training
            model.train()
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = self._validate_epoch(model, val_loader, loss_fn)
            val_losses.append(val_loss)
            
            # Update learning rate and check early stopping
            scheduler.step()# 
            current_lr = scheduler.get_last_lr()[0]
            early_stopping(val_loss, model)
            
            epochs.set_postfix_str(f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
            
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

    def _train_epoch(self, model, train_loader, optimizer, loss_fn):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            loss = loss_fn(answer, guess)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, loss_fn):
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                given = batch["given"].cuda()
                answer = batch["answer"].cuda()
                guess = model(given)
                loss = loss_fn(answer, guess)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _save_model_summary(self, model, dataset):
        sample = dataset[0]
        input_size = (self.model_config.batch_size, self.model_config.window_given, sample['given'].shape[-1])
        model_summary = summary(model, input_size=input_size, device='cuda')
        
        with open(self.model_dir / f'model_summary_{self.model_type}_{self.timestamp}.txt', 'w') as f:
            f.write(f"Model Configuration ({self.model_type}):\n")
            f.write(str(vars(self.model_config)))
            f.write("\n\nModel Summary:\n")
            f.write(str(model_summary))

    def test(self, model_path, experiment_name='', find_threshold=False):
        # 데이터 준비
        data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)
        preprocessor = HAIPreprocessor()
        valid_columns = data['test'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
        data['train'] = data['train'][data['train']['attack']!=1] # 정상데이터만 학습에 활용
        preprocessor.fit(data['train'], valid_columns)  # train 데이터로 fit
        test_processed = preprocessor.transform(data['test'], valid_columns)

        # 모델 로드
        saved_model = torch.load(model_path)
        model = self.create_model(test_processed.shape[-1])
        model.load_state_dict(saved_model["state"])
        model.eval()
        
        # 테스트 데이터셋 생성
        test_dataset = HAIDataset(
            data['test'][Config.TIMESTAMP_FIELD],
            test_processed,
            window_size=self.model_config.window_size,
            window_given=self.model_config.window_given,
            stride=1,
            attacks=data['test'][Config.ATTACK_FIELD]
        )
 
        # 예측 및 평가
        check_ts, check_dist, check_att,  predictions = self._get_predictions(model, test_dataset)
        results = self._evaluate_predictions(check_ts, predictions, check_att, data['test'], experiment_name, find_threshold=find_threshold)
        
        return results

    def _get_predictions(self, model, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.model_config.batch_size)
        predictions = []
        ts, dist, att = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                given = batch["given"].cuda()
                answer = batch["answer"].cuda()
                guess = model(given)
                ts.append(np.array(batch["ts"]))
                dist.append(torch.abs(answer - guess).cpu().numpy())
                att.append(np.array(batch["attack"]))
                predictions.append(guess.cpu().numpy())
        return (
            np.concatenate(ts),
            np.concatenate(dist),
            np.concatenate(att),
            np.concatenate(predictions)
        )        

    def _evaluate_predictions(self, check_ts, predictions, check_att, test_data, experiment_name, find_threshold=False):
        # suffix_timestamp = experiment_name.split('_')[-2:]
        def fill_blank(check_ts, labels, total_ts):
            """윈도우 방식의 예측에서 빈 구간을 채우는 함수"""
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
        
        # 이상 점수 계산
        anomaly_scores = np.mean(np.abs(predictions), axis=1)
        
        # Threshold 최적화
        if find_threshold:
            optimal_threshold, best_f1, best_metrics = self.threshold_optimizer.find_optimal_threshold(
                anomaly_scores=anomaly_scores,
                true_labels=check_att,
                start=0.0,
                end=1.0,
                steps=100
            )
            print(f"Optimal threshold: {optimal_threshold:.6f}")
        else:
            optimal_threshold = self.model_config.threshold
        
        # 예측 라벨 생성
        predicted_labels = (anomaly_scores > optimal_threshold).astype(int)
        
        # 빈 구간 채우기
        timestamps = test_data[Config.TIMESTAMP_FIELD]
        final_labels = fill_blank(check_ts, predicted_labels, timestamps)

        final_attack_labels = np.array(test_data[Config.ATTACK_FIELD])

        # 최종 TaPR 평가 수행
        tapr = etapr.evaluate(anomalies=final_attack_labels, predictions=final_labels)     
        # 상세 결과를 CSV로 저장
        results_df = pd.DataFrame({
            'timestamp': check_ts,
            'anomaly_score': anomaly_scores,
            'predicted_label': predicted_labels,
            'ground_truth': check_att
        })
        results_df.to_csv(self.result_dir / f"detailed_results_{experiment_name}_{self.timestamp}.csv", index=False)
        
        # 메트릭스를 CSV로 저장
        metrics_df = pd.DataFrame({
            'metric': ['F1', 'TaP', 'TaR', 'Detected_Anomalies'],
            'value': [tapr['f1'], tapr['TaP'], tapr['TaR'], len(tapr['Detected_Anomalies'])]
        })
        metrics_df.to_csv(self.result_dir / f"metrics_{experiment_name}_{self.timestamp}.csv", index=False)
        
        # 결과 출력 및 저장
        # 결과 파일에 threshold 정보 추가
        result_file = self.result_dir / f"test_results_{experiment_name}_{self.timestamp}.txt"
        with open(result_file, 'w') as f:
            f.write("Test Results:\n")
            f.write(f"Experiment name: {experiment_name}\n")
            f.write(f"Threshold: {optimal_threshold:.6f}\n")
            f.write(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})\n")
            f.write(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}\n")
            f.write(f"\nDetected anomalies: {tapr['Detected_Anomalies']}")
        
        print("\nTest Results:")
        print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")
        print(f"Number of detected anomalies: {len(tapr['Detected_Anomalies'])}")
        print(f"Results saved in: {self.result_dir}")
     
        return {
            'anomaly_score': anomaly_scores,
            'predictions': final_labels,
            'ground_truth': final_attack_labels,
            'metrics': tapr,
            'threshold': optimal_threshold
        }           


    def prepare_data(self):
        # 데이터 로드 및 전처리
        data = load_dataset(Config.DATA_DIR, Config.HAI_VERSION)

        preprocessor = HAIPreprocessor()
        data['train'] = data['train'][data['train']['attack']!=1]
        valid_columns = data['train'].columns.drop([Config.TIMESTAMP_FIELD, Config.ATTACK_FIELD] + Config.USELESS_FIELDS)
        preprocessor.fit(data['train'], valid_columns)
        train_processed = preprocessor.transform(data['train'], valid_columns)
        
        # 데이터셋 생성
        train_dataset = HAIDataset(
            data['train'][Config.TIMESTAMP_FIELD],
            train_processed,
            window_size=self.model_config.window_size,
            window_given=self.model_config.window_given,
            stride=100
        )
        
        return train_dataset, train_processed.shape[-1]

    def run_experiment(self, experiment_name):
        # 데이터 준비
        train_dataset, n_features = self.prepare_data()
        
        # 모델 생성
        model = self.create_model(n_features)
        
        # 학습
        best_model = self.train(train_dataset, model)

        with open(self.result_dir / f"train_results_{experiment_name}_{self.timestamp}.txt", 'w') as f:
            f.write(f"Training Configuration:\n")
            for key, value in self.model_config.__dict__.items():
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
        plt.savefig(self.result_dir / f"loss_history_{experiment_name}_{self.timestamp}.png")
        plt.close()          
        # 모델 저장
        model_path = self.model_dir / f"{self.model_type}_model_{experiment_name}_{self.timestamp}.pt"
        torch.save({
            "state": best_model["state"],
            "best_epoch": best_model["best_epoch"],
            "loss_history": best_model["loss_history"],
            "config": self.model_config.__dict__
        }, model_path)
        

        return model_path
    
def main():
    # 실험 설정
    experiment_configs = {
        'transformer_batch128_w10': {
            'config': create_model_config('transformer',
                window_size=10,
                window_given=9,
                d_model=64,
                nhead=8,
                n_layers=3,
                batch_size=128,
                n_epochs=10,
                learning_rate=0.0005,
                dropout=0.2
            ),
            'name': 'transformer_batch128_w10',
            'model_type': 'transformer'
        },        
      
        'lstm_batch64_w10': {
             'config': create_model_config('lstm',
                 window_size=10,
                 window_given=9,
                 n_hiddens=50,
                 n_layers=3,
                 batch_size=64,
                 n_epochs=300,
                 learning_rate=0.001,
                 dropout=0.2,
                 bidirectional=True
             ),
             'name': 'lstm_batch64_w10',
             'model_type': 'lstm'
         },
         'transformer_batch128_layer5_w10': {
             'config': create_model_config('transformer',
                 window_size=10,
                 window_given=9,
                 d_model=64,
                 nhead=8,
                 n_layers=5,
                 batch_size=128,
                 n_epochs=500,
                 learning_rate=0.0005,
                 dropout=0.2
             ),
             'name': 'transformer_batch128_layer5_w10',
             'model_type': 'transformer'
         },    
         'transformer_batch128_nhead16_w10': {
             'config': create_model_config('transformer',
                 window_size=10,
                 window_given=9,
                 d_model=64,
                 nhead=16,
                 n_layers=3,
                 batch_size=128,
                 n_epochs=500,
                 learning_rate=0.0005,
                 dropout=0.2
             ),
             'name': 'transformer_batch128_nhead16_w10',
             'model_type': 'transformer'
         },                          
    }
    results = []
    for exp_key, exp_config in experiment_configs.items():
        print(f"\nStarting {exp_config['model_type'].upper()} experiment: {exp_config['name']}")
        trainer = ModelTrainer(exp_config['config'], model_type=exp_config['model_type'])  # 실제 모델 타입 전달
        model_paths = trainer.run_experiment(exp_config['name'])
        result = trainer.test(model_paths, exp_config['name'], find_threshold=True)

        # 결과 저장
        results.append({
            'experiment_name': exp_config['name'],
            'config': exp_config,
            'metrics': result['metrics'],
            'threshold': result['threshold'],
            'model_path': model_paths
        })              
        print(f"{exp_config['name']} experiment completed")
        
    # 결과 요약
    print("\nExperiment Summary:")
    summary_df = pd.DataFrame([
        {
            'Experiment': r['experiment_name'],
            'Threshold': r['threshold'],
            'F1': r['metrics']['f1'],
            'TaP': r['metrics']['TaP'],
            'TaR': r['metrics']['TaR'],
            'Model Path': r['model_path']
        } for r in results
    ])
    print(summary_df)
    
    # 결과를 CSV로 저장
    timestamp_final = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df.to_csv(f'results/experiment_summary_combined_{timestamp_final}.csv', index=False)
    
if __name__ == "__main__":
    main()  


