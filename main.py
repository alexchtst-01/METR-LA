import torch
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy as dc

def createLookbackData(data, lookback):
    df = dc(data)
    col = df.columns
    
    for i in range(1, lookback+1):
        df[f'{col[1]}(t-1)'] = df[f'{col[1]}'].shift(i)
    
    df.dropna(inplace=True)
    
    return df

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_stacked_layers):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.num_stacked_layers = num_stacked_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)
    
    def forward(self, x):
        pass