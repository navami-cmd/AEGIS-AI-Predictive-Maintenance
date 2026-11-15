import os
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Configuration
# -----------------------------
WINDOW_SIZE = 30
MODEL_PATH = "data/processed/lstm_rul_model.pth"
USEFUL_SENSORS_PATH = "data/processed/useful_sensors.txt"
X_TEST_PATH = "data/processed/X_val.npy"


# -----------------------------
# LSTM Model
# -----------------------------
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
        return self.fc(last_output).squeeze(1)


# -----------------------------
# Load sensors
# -----------------------------
with open(USEFUL_SENSORS_PATH, "r") as f:
    sensors = [line.strip() for line in f if line.strip()]


# -----------------------------
# Load model
# -----------------------------
model = LSTMModel(len(sensors))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# -----------------------------
# Load validation data
# -----------------------------
X_val = np.load(X_TEST_PATH)

# Use one sample
sample = X_val[0].copy()

x = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0)


# -----------------------------
# Original prediction
# -----------------------------
with torch.no_grad():
    original_prediction = model(x).item()


# -----------------------------
# Sensor occlusion XAI
# -----------------------------
importance = []

for sensor_index, sensor_name in enumerate(sensors):

    modified = sample.copy()

    # Replace this sensor's complete 30-cycle sequence
    # with its average value
    sensor_mean = np.mean(modified[:, sensor_index])

    modified[:, sensor_index] = sensor_mean

    modified_tensor = torch.tensor(
        modified,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        new_prediction = model(modified_tensor).item()

    contribution = abs(original_prediction - new_prediction)

    importance.append({
        "sensor": sensor_name,
        "importance": contribution,
        "original_prediction": original_prediction,
        "modified_prediction": new_prediction
    })


# -----------------------------
# Sort by importance
# -----------------------------
importance.sort(
    key=lambda x: x["importance"],
    reverse=True
)


# -----------------------------
# Display results
# -----------------------------
print("\n==============================")
print("XAI SENSOR CONTRIBUTION")
print("==============================")

print(f"\nOriginal Predicted RUL: {original_prediction:.2f} cycles")

print("\nTop contributing sensors:")

for item in importance[:10]:
    print(
        f"{item['sensor']:10s} | "
        f"Importance: {item['importance']:.4f} | "
        f"Modified Prediction: {item['modified_prediction']:.2f}"
    )

# -----------------------------
# Save results
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

with open(
    "data/processed/xai_results.txt",
    "w"
) as f:

    f.write("XAI SENSOR CONTRIBUTION\n")
    f.write("=======================\n\n")

    f.write(
        f"Original Predicted RUL: "
        f"{original_prediction:.2f} cycles\n\n"
    )

    for item in importance:

        f.write(
            f"{item['sensor']} | "
            f"Importance: {item['importance']:.4f} | "
            f"Modified Prediction: "
            f"{item['modified_prediction']:.2f}\n"
        )

print("\nXAI results saved to:")
print("data/processed/xai_results.txt")