import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Data Collection
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data['Close'].values
    return stock_data

# Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.linear = nn.Linear(1, 50)  # Adjusted input dimension to 1
        self.relu = nn.ReLU()
        self.output = nn.Linear(50, 1)

    def forward(self, t, z):
        z = self.relu(self.linear(z))
        z = self.output(z)
        return z

class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, z0, t):
        z = odeint(self.ode_func, z0, t)
        return z

# Training the model
def train_model(model, data, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        t = torch.linspace(0, 1, len(data))
        z0 = torch.tensor([[data[0]]], dtype=torch.float32)  # Adjusted shape of z0
        z_pred = model(z0, t)

        loss = loss_fn(z_pred.squeeze(), torch.tensor(data, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

# Making predictions
def predict(model, initial_value, steps):
    model.eval()
    t = torch.linspace(0, steps, steps + 1)
    z0 = torch.tensor([[initial_value]], dtype=torch.float32)  # Adjusted shape of z0
    z_pred = model(z0, t)
    return z_pred.detach().numpy().squeeze()

# Visualization
def plot_results(actual, predicted):
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    stock_data = get_stock_data(ticker, start_date, end_date)

    ode_func = ODEFunc()
    model = NeuralODE(ode_func)

    trained_model = train_model(model, stock_data)
    predictions = predict(trained_model, stock_data[-1], len(stock_data) // 10)

    plot_results(stock_data, predictions)
