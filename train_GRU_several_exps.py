import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config.config_hai import create_model_config
from train_GRU import main, test
import pandas as pd
from datetime import datetime
        # adamw 기본 셋팅 lr=1e-3,
        # betas=(0.9, 0.999),
        # eps=1e-8,
        # weight_decay=1e-2,
        # amsgrad=False,
def create_experiment_configs():
    experiment_configs = [
        {
            'model_type': 'gru',
            'config': create_model_config('gru',
                window_size=10,
                window_given=9,
                n_hiddens=100,
                n_layers=3,
                batch_size=256,
                n_epochs=100,
                learning_rate=0.001,
                dropout=0.2
            ),
            'name': 'batch256_w10'
        },
        {
            'model_type': 'gru',
            'config': create_model_config('gru',
                window_size=10,
                window_given=9,
                n_hiddens=100,
                n_layers=3,
                batch_size=128,
                n_epochs=100,
                learning_rate=0.001,
                dropout=0.2
            ),
            'name': 'batch128_w10'
        },
        {
            'model_type': 'gru',
            'config': create_model_config('gru',
                window_size=10,
                window_given=9,
                n_hiddens=100,
                n_layers=3,
                batch_size=64,
                n_epochs=100,
                learning_rate=0.001,
                dropout=0.2
            ),
            'name': 'batch64_w10'
        },     
        {
            'model_type': 'gru',
            'config': create_model_config('gru',
                window_size=10,
                window_given=9,
                n_hiddens=200,
                n_layers=3,
                batch_size=128,
                n_epochs=300,
                learning_rate=0.001,
                dropout=0.2
            ),
            'name': 'batch128_h200_w10'
        },
        {
            'model_type': 'gru',
            'config': create_model_config('gru',
                window_size=10,
                window_given=9,
                n_hiddens=200,
                n_layers=3,
                batch_size=128,
                n_epochs=300,
                learning_rate=0.001,
                dropout=0.2
            ),
            'name': 'batch64_h200_w10'
        }                     
    ]
    return experiment_configs

def run_experiments():
    configs = create_experiment_configs()
    results = []
    
    for config in configs:
        print(f"\nStarting experiment: {config['name']}")
        print("Configuration:", config['config'])
        
        try:
            # 모델 학습
            model_path, experiment_name = main(
                model_config=config['config'],
                experiment_name=config['name'],
                model_type=config['model_type']
            )

            # 문자열로 변환하여 전달
            if not isinstance(model_path, (str, Path)):
                model_path = str(model_path)
            
            # 테스트 수행
            test_results = test(model_path, experiment_name, find_threshold=True)
            
            # 결과 저장
            results.append({
                'experiment_name': config['name'],
                'config': config['config'],
                'metrics': test_results['metrics'],
                'threshold': test_results['threshold'],
                'model_path': model_path
            })
       
            print(f"Experiment {config['name']} completed successfully")
            print(f"Threshold: {test_results['threshold']:.3f}")
            print(f"F1: {test_results['metrics']['f1']:.3f}")
            print(f"TaP: {test_results['metrics']['TaP']:.3f}")
            print(f"TaR: {test_results['metrics']['TaR']:.3f}")
            
        except Exception as e:
            print(f"Error in experiment {config['name']}: {str(e)}")
            continue
    
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df.to_csv(f'results/gru/experiment_summary_{configs[0]["name"]}_{timestamp}.csv', index=False)
    
    return results

if __name__ == "__main__":
    results = run_experiments()
