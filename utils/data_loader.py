import torch
from torch.utils.data import Dataset
import numpy as np
from datetime import timedelta
import dateutil.parser
from pathlib import Path
import pandas as pd

def load_dataset(data_dir, version="HAI 1.0/"):
    """데이터셋 로드"""
    train_files = sorted([x for x in Path(data_dir + version + "train-dataset/").glob("*.csv")])
    test_files = sorted([x for x in Path(data_dir + version + "test-dataset/").glob("*.csv")])
    train_data = pd.concat([pd.read_csv(x).rename(columns=lambda x: x.strip()) for x in train_files])
    test_data = pd.concat([pd.read_csv(x).rename(columns=lambda x: x.strip()) for x in test_files])
    # 특정 시간 제외
    # 특정 시간 제외
    attack_times = [
        {'name': 'sa14', 'date': '191030', 'start_time': '15:35', 'duration': 180},
        {'name': 'sa13', 'date': '191030', 'start_time': '16:33', 'duration': 154},
        {'name': 'sa12', 'date': '191030', 'start_time': '14:30', 'duration': 370},
        {'name': 'sa11', 'date': '191031', 'start_time': '08:42', 'duration': 348},
        {'name': 'ma01', 'date': '191031', 'start_time': '10:30', 'duration': 518},
        {'name': 'ma02', 'date': '191031', 'start_time': '11:33', 'duration': 346},
        {'name': 'ma03', 'date': '191031', 'start_time': '14:30', 'duration': 396},
        {'name': 'ma04', 'date': '191031', 'start_time': '15:41', 'duration': 348},
        {'name': 'ma05', 'date': '191031', 'start_time': '16:29', 'duration': 398},
        {'name': 'ma06', 'date': '191101', 'start_time': '09:29', 'duration': 560},
        {'name': 'ma07', 'date': '191101', 'start_time': '10:41', 'duration': 310},
        {'name': 'sa14', 'date': '191101', 'start_time': '11:23', 'duration': 180},
        {'name': 'ma11', 'date': '191101', 'start_time': '17:20', 'duration': 410},
        {'name': 'ma13', 'date': '191104', 'start_time': '17:20', 'duration': 520},
        {'name': 'ma14', 'date': '191105', 'start_time': '09:30', 'duration': 380},
        {'name': 'ma15', 'date': '191105', 'start_time': '10:20', 'duration': 290},
        {'name': 'sa11', 'date': '191105', 'start_time': '11:23', 'duration': 340},
        {'name': 'ma16', 'date': '191105', 'start_time': '12:30', 'duration': 2880},
        {'name': 'ma17', 'date': '191105', 'start_time': '14:45', 'duration': 2880}
    ]

    # 데이터 추출 - train1 (9/15) train2 (11/1~4)
    train_data_out,extracted_periods_train = extract_attack_periods(train_data, attack_times)
    test_data_out,extracted_periods_test = extract_attack_periods(test_data, attack_times)

    return {
        'train': train_data_out,
        'test': test_data_out
    }
def extract_attack_periods(data, attack_times):
    """
    특정 공격 시간대의 데이터를 추출하는 함수
    
    Parameters:
    - data: DataFrame (time 컬럼이 있어야 함)
    - attack_times: 공격 시간 정보 리스트
        [
            {
                'name': 'sa14',
                'date': '191030',
                'start_time': '15:35',
                'duration': 180
            },
            ...
        ]
    
    Returns:
    - extracted_data: 각 공격 시간대별로 추출된 데이터를 포함하는 딕셔너리
    """
    extracted_data = {}
    data_out = data.copy()
    convert_time_o = pd.to_datetime(data['time'])
    for attack in attack_times:
        # datetime 객체 생성
        start_datetime = pd.to_datetime(f"20{attack['date']} {attack['start_time']}")
        end_datetime = start_datetime + pd.Timedelta(seconds=attack['duration'])
        convert_time = pd.to_datetime(data_out['time'])
        condition1 = (convert_time < end_datetime)
        condition2 = (convert_time >= start_datetime)
        # 해당 시간대의 데이터 추출
        mask = condition1 & condition2
        mask_o = (convert_time_o >= start_datetime) & (convert_time_o< end_datetime)
        data_out = data_out[~mask]
        # 실제 데이터 확인
        if (condition1 & condition2).sum() == 0:

            mask_around = (convert_time >= (start_datetime - pd.Timedelta(minutes=1))) & \
                         (convert_time <= (end_datetime + pd.Timedelta(minutes=1)))
            #print(data[mask_around]['time'].sort_values())
                
        extracted_data[attack['name']] = data[mask_o].copy()

    return data_out,extracted_data
class HAIDataset(Dataset):
    def __init__(self, timestamps, df, window_size=90, window_given=89, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.window_size = window_size
        self.window_given = window_given
        
        self.valid_idxs = [] # 윈도우 내의 첫 시간과 마지막 시간의 차이가 정확히 window_size - 1초여야 함
        for L in range(len(self.ts) - window_size + 1): 
            R = L + window_size - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L]) == timedelta(seconds=window_size - 1):
                self.valid_idxs.append(L)
                
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        
        self.attacks = np.array(attacks, dtype=np.float32) if attacks is not None else None
        self.with_attack = attacks is not None

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + self.window_size - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + self.window_given])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        
        return item 