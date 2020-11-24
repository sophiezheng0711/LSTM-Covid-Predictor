import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def preprocess(raw, lookback):
    xs = []
    ys = []
    for i in range(len(raw) - lookback - 1):
        x = raw[i:(i+lookback)]
        y = raw[i+lookback]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class CovidPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CovidPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.dense = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.dense(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out


if __name__ == "__main__":
    df = pd.read_csv("./data/time_series_covid19_confirmed_US.csv")
    df = df.iloc[:, 11:]
    df.isnull().sum().sum()
    daily_cases = df.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index)
    daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
    daily_cases.head()

    test_data_size = 60
    lookback = 4
    train_data = daily_cases[:-test_data_size]
    test_data = daily_cases[-test_data_size:]
    scaler = MinMaxScaler()
    scaler = scaler.fit(np.expand_dims(train_data, axis=1))
    train_data = scaler.transform(np.expand_dims(train_data, axis=1))
    test_data = scaler.transform(np.expand_dims(test_data, axis=1))

    x_train, y_train = preprocess(train_data, lookback)
    x_test, y_test = preprocess(test_data, lookback)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    epochs = 100

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = CovidPredictor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

    loss_fn = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    training_pred = []

    # training
    for epoch in range(epochs):
        output = model(x_train)
        training_pred = output.detach().numpy()
        loss = loss_fn(output, y_train)
        if epoch % 10 == 0 and epoch !=0:
            print("Epoch ", epoch, "MSE: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # testing
    test_pred = scaler.inverse_transform(model(x_test).detach().numpy())
    test_plot = scaler.inverse_transform(y_test.detach().numpy())
    train_pred = scaler.inverse_transform(training_pred)
    train_plot = scaler.inverse_transform(y_train.detach().numpy())

    # plot
    plt.plot(np.concatenate((train_pred, test_pred)), label="prediction")
    plt.plot(np.concatenate((train_plot, test_plot)), label="data")
    plt.axvline(x=len(x_train), c='r', linestyle='--')
    plt.legend()
    plt.show()