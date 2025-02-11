## Create_models

#-*- coding: utf-8 -*-
# 대부분의 코드는 Baseline을 따랐으며, 수정된 부분만 주석처리하여 설명함.
import sys
from pathlib import Path
from datetime import timedelta
import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange
from TaPR_pkg import etapr
from contextlib2 import redirect_stdout
import random

# Feature Test 중 Seed 고정을 위해 사용하였음.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

# 학습과정에서 Optimizer는 Baseline코드와 동일하며, L1Loss(MAE)로 변경하여 사용함
# 데이터 shuffle은 유지하였음
def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())# ------------------- Check 
    loss_fn = torch.nn.L1Loss()# ------------------- Check 
    epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            guess = model(given)
            answer = batch["answer"].cuda()
            loss = loss_fn(answer, guess)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
    return best, loss_history

class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in trange(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False
    def __len__(self):
        return self.n_idxs
    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item

# Relu함수를 쓰기 위해 코드 추가하였음 "------------------- Check " << 부분
class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)
        self.relu = torch.nn.ReLU()# ------------------- Check 

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1]))# ------------------- Check 
        #out = self.fc(outs[-1])
        return x[0] + out

# Switch 파라미터를 통해서 검증그래프와 테스트그래프를 저장함
def check_graph(xs, att, piece=2, THRESHOLD=None, Switch=1):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        axs[i].set_ylim(0, 0.2)
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD!=None:
            axs[i].axhline(y=THRESHOLD, color='r')
    if Switch == 1:
        plt.savefig("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + '_' + str(THRESHOLD) + '_' + 'evaluation.png')
    elif Switch == 2:
        plt.savefig("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + '_' + str(THRESHOLD) + '_' + 'baseline.png')


def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

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
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(batch_size))

    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )

# Data Load & Merge
TRAIN_DATASET = sorted([x for x in Path("data/HAI 2.0/training/").glob("*.csv")]) # ------------------ Check 
TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)

VALIDATION_DATASET = sorted([x for x in Path("data/HAI 2.0/validation/").glob("*.csv")])# ------------------ Check 
VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)

TEST_DATASET = sorted([x for x in Path("data/HAI 2.0/testing/").glob("*.csv")])# ------------------ Check 
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)


DROP_FIELD = ["time", 
              "C02", "C03", "C14", "C18", "C19", "C21", "C22", "C25", "C33", "C34", "C35", "C37", "C40", "C43", "C51", "C52", "C59", "C61", "C63", "C64", "C65", "C67",
              "C04", "C05", "C06", "C07", "C08", "C10", "C11", "C17", "C24", "C28", "C32", "C44", "C46", "C48", "C49", "C50", "C53", "C58", "C62", "C71", "C76", "C78", "C79"]
TIMESTAMP_FIELD = "time"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop(DROP_FIELD) # DROP_FIELD를 통해 normalization에 사용하지 않을 변수를 제거함.
TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

# Min-Max Normalize
TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

# Boundary Check
print(boundary_check(TRAIN_DF))
print(boundary_check(TEST_DF))
# WINDOW_GIVEN_LIST 변수와 FOR문을 통해서 Window size가 다른 각 각의 모델을 생성하였음
# 모델의 THRESHOLD가 0.04 수준까지 되었을 때(오탐이 없는), 좋은 모델이라 판단하였음
# 즉, 1번의 모델생성으로 좋은 모델이 생성되기 어렵다. 
# > 모든 seed를 고정시켜 테스팅하는 것은 많은 비용이 발생하기 때문에, 그리고 모델의 랜덤성을 주기 위해 반복적으로 모델을 생성하여 좋은 모델을 선택(각 모델이 상호 보완 될 수 있도록)하였음
WINDOW_GIVEN_LIST = [39, 44, 49, 54, 59, 64, 69, 74, 79]
# Model Config
for i in range(len(WINDOW_GIVEN_LIST)):
    Filenum = "E(50)_"
    N_HIDDENS = 200 # 
    N_LAYERS = 3 # 
    #EPOCH = 50 # 
    EPOCH = 1 # 
    WINDOW_GIVEN = WINDOW_GIVEN_LIST[i]
    BATCH_SIZE = 2024
    WINDOW_SIZE = WINDOW_GIVEN + 1
    THRESHOLD = 0.04

    HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=1)

    MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
    MODEL.cuda()

    # Find Best Model
    MODEL.train()
    BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, EPOCH)
    BEST_MODEL["loss"], BEST_MODEL["epoch"]

    with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + "_" + str(N_HIDDENS) + "_" + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_model.pt", "wb") as f:
        torch.save(
            {
                "state": BEST_MODEL["state"],
                "best_epoch": BEST_MODEL["epoch"],
                "loss_history": LOSS_HISTORY,
            },
            f,
        )
    with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + "_" + str(N_HIDDENS) + "_" + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_model.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)

    torch.cuda.empty_cache()

    MODEL.load_state_dict(SAVED_MODEL["state"])

    plt.figure(figsize=(16, 4))
    plt.title("Training Loss Graph")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(SAVED_MODEL["loss_history"])
    plt.savefig("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_loss.png")
    with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_loss_history.txt", 'w') as f:
        with redirect_stdout(f):
            print(SAVED_MODEL["loss_history"])

    # Validation Check
    HAI_DATASET_VALIDATION = HaiDataset(VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD])

    # Model Load
    with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_model.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)

    MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
    MODEL.load_state_dict(SAVED_MODEL["state"])
    MODEL.to(torch.device("cuda")) # use GPU 

    MODEL.eval()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, MODEL, BATCH_SIZE)

    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
    check_graph(ANOMALY_SCORE, CHECK_ATT, piece=3, THRESHOLD=THRESHOLD, Switch=1)

    LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
    LABELS, LABELS.shape

    ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
    ATTACK_LABELS, ATTACK_LABELS.shape

    FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))
    FINAL_LABELS.shape

    ATTACK_LABELS.shape[0] == FINAL_LABELS.shape[0]

    # 검증데이터 평가결과 저장
    TaPR = etapr.evaluate(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
    with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + '_'+ str(THRESHOLD) + '_evaluation.txt', 'w') as f:
        with redirect_stdout(f):
            print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
            print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
            print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

    torch.cuda.empty_cache()

    # HAI_DATASET_TEST = HaiDataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None)

    # # Model Load
    # with open("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + "_model.pt", "rb") as f:
    #     SAVED_MODEL = torch.load(f)

    # MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
    # MODEL.load_state_dict(SAVED_MODEL["state"])
    # MODEL.to(torch.device("cuda")) # use GPU 

    # MODEL.eval() 
    # CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_TEST, MODEL, BATCH_SIZE)

    # ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
    # check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=THRESHOLD, Switch=2)
    # LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
    # LABELS, LABELS.shape

    # submission = pd.read_csv('data/HAI 2.0/sample_submission.csv') # ------------------ Check 
    # submission.index = submission['time']
    # submission.loc[CHECK_TS,'attack'] = LABELS
    # submission
    # submission.to_csv("output(Create_models)/" + str(Filenum) + '_' + str(WINDOW_GIVEN) + '_' + str(N_HIDDENS) + '_' + str(N_LAYERS) + '_' + str(BATCH_SIZE) + '_'+ str(THRESHOLD) + '_baseline.csv', index=False)

    # torch.cuda.empty_cache()
    
    # del HAI_DATASET_TRAIN, HAI_DATASET_VALIDATION, HAI_DATASET_TEST
