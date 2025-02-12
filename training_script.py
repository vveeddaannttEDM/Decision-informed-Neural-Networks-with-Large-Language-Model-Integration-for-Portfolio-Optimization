import torch
import torch.nn as nn
import torch.optim as optim
from dinn_model import DINN  # Import the DINN model
import pandas as pd

# Load dataset (Example: S&P 100 stocks)
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['returns'] = df['Close'].pct_change()
    df = df.dropna()
    return df

# Training function
def train_dinn(model, train_data, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data['time_series'], train_data['text'])
        loss = loss_fn(outputs, train_data['target'])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Load and preprocess data
num_assets = 30  # Example: DOW 30 stocks
train_data = {
    'time_series': torch.randn(100, num_assets, 128),  # Simulated time-series data
    'text': {"input_ids": torch.randint(0, 30522, (100, 50)), "attention_mask": torch.ones(100, 50)},  # Simulated LLM inputs
    'target': torch.randn(100, num_assets)
}

# Initialize and train model
dinn_model = DINN(num_assets)
train_dinn(dinn_model, train_data)
