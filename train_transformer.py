import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib

from transformer_model import TimeSeriesTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CMAPSS_DIR = os.path.join(BASE_DIR, 'data', 'CMAPSSData')
MODEL_PATH = os.path.join(DATA_DIR, 'transformer_rul_model.pth')
PRED_PATH = os.path.join(DATA_DIR, 'transformer_test_predictions.csv')

EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0005


def load_test_data(sensors, scaler):
    columns = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    columns += [f'sensor_{i}' for i in range(1, 22)]
    test = pd.read_csv(
        os.path.join(CMAPSS_DIR, 'test_FD001.txt'),
        sep=r'\s+', header=None, names=columns,
    )
    rul = np.loadtxt(os.path.join(CMAPSS_DIR, 'RUL_FD001.txt')).astype(np.float32)

    X, y, engines = [], [], []
    for i, engine_id in enumerate(sorted(test.unit.unique())):
        group = test[test.unit == engine_id].sort_values('cycle')
        recent = group.tail(30)
        if len(recent) < 30:
            continue
        X.append(scaler.transform(recent[sensors].values))
        y.append(float(rul[i]))
        engines.append(int(engine_id))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), engines


def main():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy')).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy')).astype(np.float32)
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy')).astype(np.float32)
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy')).astype(np.float32)

    with open(os.path.join(DATA_DIR, 'useful_sensors.txt')) as f:
        sensors = [line.strip() for line in f if line.strip()]
    scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer(X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val = float('inf')
    os.makedirs(DATA_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        train_loss = total / len(X_train)
        print(f'Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # Test evaluation
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    X_test, y_test, engines = load_test_data(sensors, scaler)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    pd.DataFrame({
        'Engine': engines,
        'Actual_RUL': y_test,
        'Predicted_RUL': pred,
    }).to_csv(PRED_PATH, index=False)

    print('\nTransformer training completed.')
    print(f'Best validation loss: {best_val:.4f}')
    print(f'Test MAE : {mae:.2f} cycles')
    print(f'Test RMSE: {rmse:.2f} cycles')
    print(f'Model saved to: {MODEL_PATH}')
    print(f'Predictions saved to: {PRED_PATH}')


if __name__ == '__main__':
    main()
