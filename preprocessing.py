import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_FILE = "data/CMAPSSData/train_FD001.txt"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. COLUMN NAMES
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
# 3. LOAD DATA
# ============================================================

print("Loading C-MAPSS FD001 training data...")

df = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Original dataset shape:", df.shape)


# ============================================================
# 4. CALCULATE RUL
# ============================================================

max_cycle = df.groupby("unit")["cycle"].max()

df["RUL"] = df.apply(
    lambda row: max_cycle[row["unit"]] - row["cycle"],
    axis=1
)

print("RUL calculated successfully.")


# ============================================================
# 5. SENSOR COLUMNS
# ============================================================

sensor_columns = [
    f"sensor_{i}"
    for i in range(1, 22)
]


# ============================================================
# 6. FIND CONSTANT SENSORS
# ============================================================

constant_sensors = []

for sensor in sensor_columns:

    if df[sensor].std() == 0:
        constant_sensors.append(sensor)

print("\nConstant sensors:")
print(constant_sensors)


# Remove constant sensors
useful_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]

print("\nUseful sensors:")
print(useful_sensors)

print("\nNumber of useful sensors:", len(useful_sensors))


# ============================================================
# 7. TRAIN / VALIDATION SPLIT BY ENGINE
# ============================================================

# IMPORTANT:
# We split complete engines instead of random rows.
# This prevents information from the same engine
# appearing in both training and validation data.

units = df["unit"].unique()

np.random.seed(42)

np.random.shuffle(units)

split_index = int(len(units) * 0.8)

train_units = units[:split_index]
val_units = units[split_index:]

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()

print("\nTraining engines:", len(train_units))
print("Validation engines:", len(val_units))

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))


# ============================================================
# 8. SCALE SENSOR DATA
# ============================================================

scaler = MinMaxScaler()

# Fit ONLY on training data
train_df[useful_sensors] = scaler.fit_transform(
    train_df[useful_sensors]
)

# Apply the same scaler to validation data
val_df[useful_sensors] = scaler.transform(
    val_df[useful_sensors]
)

print("\nSensor scaling completed.")


# ============================================================
# 9. CREATE TIME-SERIES WINDOWS
# ============================================================

WINDOW_SIZE = 30


def create_sequences(data, sensors, window_size):

    X = []
    y = []

    for unit in data["unit"].unique():

        engine = data[data["unit"] == unit].sort_values("cycle")

        sensor_values = engine[sensors].values
        rul_values = engine["RUL"].values

        for i in range(len(engine) - window_size + 1):

            sequence = sensor_values[
                i:i + window_size
            ]

            target = rul_values[
                i + window_size - 1
            ]

            X.append(sequence)
            y.append(target)

    return np.array(X), np.array(y)


print("\nCreating training sequences...")

X_train, y_train = create_sequences(
    train_df,
    useful_sensors,
    WINDOW_SIZE
)

print("Creating validation sequences...")

X_val, y_val = create_sequences(
    val_df,
    useful_sensors,
    WINDOW_SIZE
)


# ============================================================
# 10. DISPLAY SHAPES
# ============================================================

print("\n==============================")
print("FINAL DATA SHAPES")
print("==============================")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train
)

np.save(
    f"{OUTPUT_DIR}/X_val.npy",
    X_val
)

np.save(
    f"{OUTPUT_DIR}/y_val.npy",
    y_val
)


# Save scaler parameters
import joblib

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}/scaler.pkl"
)


# Save useful sensor names
with open(
    f"{OUTPUT_DIR}/useful_sensors.txt",
    "w"
) as file:

    for sensor in useful_sensors:
        file.write(sensor + "\n")


print("\nProcessed files saved successfully!")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nPreprocessing pipeline completed!")