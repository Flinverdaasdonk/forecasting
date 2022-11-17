import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[16, 8], output_size=1):
        super().__init__()
        self.non_linearity = nn.Tanh


        if not isinstance(hidden_layer_sizes, list):
            assert isinstance(hidden_layer_sizes, int)
            hidden_layer_sizes = [hidden_layer_sizes]

        layer_sizes = hidden_layer_sizes + [output_size]

        self.init_hidden_cell = (torch.zeros(1,1,layer_sizes[0]),
                    torch.zeros(1,1,layer_sizes[0]))

        self.hidden_cell = self.init_hidden_cell

        self.lstm = nn.LSTM(input_size, layer_sizes[0])

        self.flatten = nn.Flatten(end_dim=-1)

        linear_layers = []

        for ls0, ls1 in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear_layers.append(self.non_linearity())
            linear_layers.append(nn.Linear(ls0, ls1))

        self.layer_sizes = layer_sizes
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, input_seq):
        self.hidden_cell  = self.init_hidden_cell
    
        x, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        x = self.flatten(x)

        x = self.linear(x)
        return x[-1]

def df_to_X_and_y(df):
    y_series = df["y"]
    X_df = df.drop(columns="y") 

    if "datetimes" in X_df.columns:
        X_df = X_df.drop(columns="datetimes")

    X = torch.FloatTensor(X_df.to_numpy())
    y = torch.FloatTensor(y_series.to_numpy())

    return X, y


def create_inout_sequences(X, y, tw):
    assert len(X) == len(y)

    inout_seq = []
    L = len(y)

    for i in range(L-tw+1):
        X_seq = X[i:i+tw]
        y_seq = y[i+tw-1:i+tw]
        inout_seq.append((X_seq , y_seq))
    return inout_seq


def lazy_create_inout_sequences(df, tw):
    X, y = df_to_X_and_y(df)
    inout_seq = create_inout_sequences(X, y, tw)

    return inout_seq
