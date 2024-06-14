import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fully_connected = nn.Linear(hidden_size, 1)
        
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fully_connected(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        super(TimeSeriesDataset, self).__init__()
        self.seq = torch.reshape(torch.tensor(X), (X.shape[0], x.shape[1])).float()
        self.out = torch.reshape(torch.tensor(y), (y.shape[0], y.shape[1])).float()
    
    def __getitem__(self, index):
        return self.seq[index], self.out[index]

    def __len__(self):
        return len(self.seq)

def data_with_lookback(df, n_steps):
    df = dc(df)
    col = df.columns
    df.set_index('Time', inplace=True)
    for i in range(n_steps + 1):
        df[f"{col[1]}(t-{i})"] = df[f"{col[1]}"] .shift(i)
    
    df.dropna(inplace=True)
    return df

def TrainEval(model, criterion, optimizer, fold):
    for epoch in range(EPOCHS):
        model_general.train(True)
        train_running_loss = 0.0
        for idx, data in enumerate(train_loader):
            seq = data[0].float() #.to(device)
            out = data[1].float() #.to(device)
            optimizer.zero_grad()
            pred = model_general(seq.unsqueeze(0))
            loss = criterion(pred, out)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_error_store.append(train_running_loss / (idx+1))

        model_general.eval()
        eval_running_loss = 0.0
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                seq = data[0].float() #.to(device)
                out = data[1].float() #.to(device)
                pred = model_general(seq.unsqueeze(0))
                loss = criterion(pred, out)
                eval_running_loss += loss.item()
            test_error_store.append(eval_running_loss / (idx+1))

        print(f"in folds: {fold+1} Epoch: {epoch+1}")
        print(f"train loss: {train_running_loss / (idx+1)}")
        print(f"test loss: {eval_running_loss / (idx+1)}")