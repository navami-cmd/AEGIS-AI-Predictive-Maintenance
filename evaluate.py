import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/processed"
CMAPSS_DIR = "data/CMAPSSData"

MODEL_PATH = f"{DATA_DIR}/lstm_rul_model.pth"
SCALER_PATH = f"{DATA_DIR}/scaler.pkl"

WINDOW_SIZE = 30


# ============================================================
# COLUMN NAMES
# ============================================================

columns = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


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

print("Useful sensors:")
print(useful_sensors)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_df = pd.read_csv(
    f"{CMAPSS_DIR}/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Test dataset shape:", test_df.shape)


# ============================================================
# LOAD TRUE RUL VALUES
# ============================================================

true_rul = pd.read_csv(
    f"{CMAPSS_DIR}/RUL_FD001.txt",
    header=None,
    names=["RUL"]
)

true_rul = true_rul["RUL"].values

print("Number of test engines:", len(true_rul))


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)


# ============================================================
# SCALE TEST SENSOR DATA
# ============================================================

test_df[useful_sensors] = scaler.transform(
    test_df[useful_sensors]
)

print("Test sensor scaling completed.")


# ============================================================
# CREATE FINAL SEQUENCE FOR EACH TEST ENGINE
# ============================================================

X_test = []
y_test = []


for unit in sorted(test_df["unit"].unique()):

    engine = test_df[
        test_df["unit"] == unit
    ].sort_values("cycle")

    sensor_values = engine[
        useful_sensors
    ].values

    # We use the final 30 cycles
    # because the official RUL value corresponds
    # to the final observed cycle of each test engine.

    if len(engine) >= WINDOW_SIZE:

        sequence = sensor_values[-WINDOW_SIZE:]

    else:

        # Padding for safety if an engine has fewer than 30 cycles

        padding = np.repeat(
            sensor_values[0:1],
            WINDOW_SIZE - len(engine),
            axis=0
        )

        sequence = np.vstack(
            [padding, sensor_values]
        )

    X_test.append(sequence)


# Convert to numpy
X_test = np.array(X_test)

y_test = true_rul


print("\nTest sequences created.")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


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
# LOAD TRAINED MODEL
# ============================================================

input_size = X_test.shape[2]

model = LSTMModel(input_size)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

model.eval()

print("\nTrained LSTM model loaded.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

with torch.no_grad():

    predictions = model(
        X_test_tensor
    ).numpy()


# ============================================================
# CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("C-MAPSS FD001 TEST RESULTS")
print("========================================")

print(f"MAE  : {mae:.2f} cycles")
print(f"RMSE : {rmse:.2f} cycles")

print("========================================")


# ============================================================
# SHOW SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Engine": range(1, len(y_test) + 1),
    "Actual_RUL": y_test,
    "Predicted_RUL": predictions
})

print("\nSample predictions:")
print(results.head(10))


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    f"{DATA_DIR}/test_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print(f"{DATA_DIR}/test_predictions.csv")