import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

class MultiOutputXGBoost:
    def __init__(self, n_features, n_models=3, learning_rate=0.01, max_depth=4, n_estimators=200):
        self.n_features = n_features
        self.n_models = n_models
        self.models = []
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': learning_rate,  # 더 작은 learning rate
            'max_depth': max_depth,  # 트리 깊이 감소
            'n_estimators': n_estimators,  # 트리 개수 증가
            'subsample': 0.8,  # 데이터 샘플링
            'colsample_bytree': 0.8,  # 특성 샘플링
            'min_child_weight': 3,  # 과적합 방지
            'gamma': 0.1,  # 트리 분할 임계값
            'reg_alpha': 0.1,  # L1 정규화
            'reg_lambda': 1.0,  # L2 정규화
        }
        
        # 각 출력 변수마다 별도의 XGBoost 모델 생성
        # for _ in range(n_features):
        #     self.models.append(xgb.XGBRegressor(**self.params))

        # 각 앙상블 모델(n_models)마다 n_features개의 XGBoost 모델 생성
        for _ in range(n_models):
            model_list = []
            for _ in range(n_features):
                model_list.append(xgb.XGBRegressor(**self.params))
            self.models.append(model_list)        
    
    def fit(self, X, y):
        # 검증 세트 분할 (20%)
        split_idx = int(len(X) * 0.8)
        
        self.best_iterations = []  # 각 모델의 최적 반복 횟수 저장
        
        for ensemble_idx, model_list in enumerate(self.models):
            print(f"\nTraining ensemble model {ensemble_idx + 1}/{self.n_models}")
            
            # 부트스트랩 샘플링
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 학습/검증 세트 분할
            X_train = X_bootstrap[:split_idx]
            X_val = X_bootstrap[split_idx:]
            y_train = y_bootstrap[:split_idx]
            y_val = y_bootstrap[split_idx:]
            
            for feature_idx, model in enumerate(model_list):
                print(f"Training feature {feature_idx + 1}/{self.n_features}...")
                
                best_val_score = float('inf')
                patience = 10  # early stopping patience
                counter = 0
                best_model = None
                
                # 점진적으로 트리를 추가하며 학습
                for n in range(50, self.params['n_estimators'] + 1, 50):  # 50개씩 트리 추가
                    # 현재 트리 개수로 모델 설정
                    current_params = self.params.copy()
                    current_params['n_estimators'] = n
                    
                    # 새로운 모델 학습
                    model = xgb.XGBRegressor(**current_params)
                    model.fit(
                        X_train, y_train[:, feature_idx],
                        eval_set=[(X_val, y_val[:, feature_idx])],
                        verbose=False
                    )
                    
                    # 검증 성능 평가
                    val_pred = model.predict(X_val)
                    val_score = mean_squared_error(y_val[:, feature_idx], val_pred)
                    
                    # 성능이 개선되었는지 확인
                    if val_score < best_val_score:
                        best_val_score = val_score
                        best_model = model
                        counter = 0
                    else:
                        counter += 1
                    
                    # Early stopping 체크
                    if counter >= patience:
                        print(f"Early stopping at {n} trees for feature {feature_idx + 1}")
                        break
                
                # 최적의 모델 저장
                model_list[feature_idx] = best_model
                print(f"Feature {feature_idx + 1} best validation RMSE: {np.sqrt(best_val_score):.6f}")
                
            # 앙상블 모델의 평균 검증 점수 계산
            ensemble_val_pred = np.column_stack([
                model.predict(X_val) for model in model_list
            ])
            ensemble_val_score = mean_squared_error(y_val, ensemble_val_pred)
            print(f"Ensemble {ensemble_idx + 1} overall validation RMSE: {np.sqrt(ensemble_val_score):.6f}")
            
    def predict(self, X):
        # 각 앙상블 모델의 예측값 계산
        ensemble_predictions = []
        
        for model_list in self.models:
            # 각 특성별 예측
            feature_predictions = []
            for model in model_list:
                pred = model.predict(X)
                feature_predictions.append(pred)
            
            # 특성별 예측을 결합
            ensemble_pred = np.column_stack(feature_predictions)
            ensemble_predictions.append(ensemble_pred)
        
        # 모든 앙상블 모델의 예측 평균
        final_prediction = np.mean(ensemble_predictions, axis=0)
        return final_prediction