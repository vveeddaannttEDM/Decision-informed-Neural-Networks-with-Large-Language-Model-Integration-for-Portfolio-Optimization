import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoModel

# Load Financial Data (Assume S&P100 dataset)
df = pd.read_csv('data/sp100.csv')
df['returns'] = df['Close'].pct_change()
df = df.dropna()

# Define LLM Embedding Layer
class LLMBasedEmbedding(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(LLMBasedEmbedding, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.llm.config.hidden_size

    def forward(self, text_inputs):
        outputs = self.llm(**text_inputs)
        return outputs.last_hidden_state[:, 0, :]

# Transformer for Time-Series Forecasting
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=n_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)

# Optimization Layer (Mean-Variance Portfolio Allocation)
class PortfolioOptimizer(nn.Module):
    def __init__(self, num_assets):
        super(PortfolioOptimizer, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_assets))
    
    def forward(self, returns, risk_aversion=0.1):
        cov_matrix = torch.cov(returns.T)
        inv_cov = torch.inverse(cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0]))  # Regularization
        optimal_weights = inv_cov @ returns.mean(dim=0) / risk_aversion
        return nn.functional.softmax(optimal_weights, dim=0)

# Full Decision-Informed Neural Network (DINN)
class DINN(nn.Module):
    def __init__(self, num_assets, input_dim=128, hidden_dim=256):
        super(DINN, self).__init__()
        self.llm_embedding = LLMBasedEmbedding()
        self.transformer = TransformerModel(input_dim, hidden_dim, num_assets)
        self.optimizer_layer = PortfolioOptimizer(num_assets)

    def forward(self, time_series_data, text_data):
        text_embedding = self.llm_embedding(text_data)
        predictions = self.transformer(time_series_data)
        portfolio_weights = self.optimizer_layer(predictions)
        return portfolio_weights

# Training and Testing
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

# Initialize and Train the Model
dinn_model = DINN(num_assets)
train_dinn(dinn_model, train_data)
