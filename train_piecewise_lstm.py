import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(DATA_DIR, 'piecewise_lstm_rul_model.pth')

RUL_CAP = 125.0
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze(1)


def main():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

    # Piecewise RUL target: cap large early-life values while preserving
    # the degradation region that matters most for maintenance decisions.
    y_train = np.minimum(y_train, RUL_CAP).astype(np.float32)
    y_val = np.minimum(y_val, RUL_CAP).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float('inf')
    os.makedirs(DATA_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        train_loss = train_sum / len(X_train)
        print(f'Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print('\nPiecewise-RUL LSTM training completed.')
    print(f'RUL cap: {RUL_CAP:.0f} cycles')
    print(f'Best validation loss: {best_val:.4f}')
    print(f'Model saved to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
