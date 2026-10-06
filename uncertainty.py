import numpy as np
import torch
import torch.nn as nn
import joblib


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/processed"

MODEL_PATH = f"{DATA_DIR}/lstm_rul_model.pth"
SCALER_PATH = f"{DATA_DIR}/scaler.pkl"

WINDOW_SIZE = 30
NUM_SAMPLES = 50


# ============================================================
# LOAD USEFUL SENSORS
# ============================================================

with open(
    f"{DATA_DIR}/useful_sensors.txt",
    "r"
) as file:

    useful_sensors = [
        line.strip()
        for line in file
        if line.strip()
    ]


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
# LOAD MODEL
# ============================================================

model = LSTMModel(
    len(useful_sensors)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)


# ============================================================
# ENABLE DROPOUT
# ============================================================

def enable_dropout(model):

    for module in model.modules():

        if isinstance(
            module,
            nn.Dropout
        ):

            module.train()


# ============================================================
# UNCERTAINTY PREDICTION
# ============================================================

def predict_with_uncertainty(
    sequence
):

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    model.eval()

    # Enable dropout during inference
    enable_dropout(model)

    predictions = []

    for _ in range(NUM_SAMPLES):

        with torch.no_grad():

            prediction = model(
                sequence
            ).item()

        predictions.append(
            prediction
        )

    predictions = np.array(
        predictions
    )

    mean_prediction = predictions.mean()

    std_prediction = predictions.std()

    lower_bound = (
        mean_prediction -
        1.96 * std_prediction
    )

    upper_bound = (
        mean_prediction +
        1.96 * std_prediction
    )

    return (
        max(0, mean_prediction),
        max(0, lower_bound),
        max(0, upper_bound),
        std_prediction,
        predictions
    )


# ============================================================
# DEMO TEST
# ============================================================

if __name__ == "__main__":

    X_val = np.load(
        f"{DATA_DIR}/X_val.npy"
    )

    sample = X_val[0]

    mean_rul, lower, upper, uncertainty, predictions = (
        predict_with_uncertainty(sample)
    )

    print("\n========================================")
    print("UNCERTAINTY ESTIMATION")
    print("========================================")

    print(
        f"Predicted RUL : {mean_rul:.2f} cycles"
    )

    print(
        f"Uncertainty   : ±{uncertainty:.2f} cycles"
    )

    print(
        f"Lower bound   : {lower:.2f} cycles"
    )

    print(
        f"Upper bound   : {upper:.2f} cycles"
    )

    print(
        f"Samples       : {NUM_SAMPLES}"
    )

    print("========================================")