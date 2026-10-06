import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/processed"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_PATH = "data/processed/lstm_rul_model.pth"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed data...")

X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")

X_val = np.load(f"{DATA_DIR}/X_val.npy")
y_val = np.load(f"{DATA_DIR}/y_val.npy")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

input_size = X_train.shape[2]

model = LSTMModel(input_size)

print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)


    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


print("\nTraining completed!")

print("Best validation loss:", best_val_loss)

print("Model saved to:")
print(MODEL_PATH)