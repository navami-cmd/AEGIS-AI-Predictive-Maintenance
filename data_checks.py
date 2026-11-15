"""
data_checks.py - input validation for the AEGIS pipeline.

prepare_window() turns one engine's history into the scaled 30-cycle model input
and reports anything unusual as warnings. It raises DataError (a ValueError) for
problems it cannot safely repair, so the dashboard can show a clear message
instead of crashing or silently predicting from bad data.

Repair rules (all reported in `warnings`):
  * rows are sorted by cycle; duplicate cycles keep the last row
  * isolated NaN / inf sensor values are forward-filled, then back-filled, within
    the engine's own history (never from another engine)
  * a sensor that is entirely missing for the engine is an error
  * engines with fewer than WINDOW_SIZE cycles are padded by repeating their first
    row (the same rule evaluate.py uses) and flagged as less reliable
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 30
SPLIT_SEED = 42            # preprocessing.py: np.random.seed(42)
TRAIN_FRACTION = 0.8       # preprocessing.py: 80% of engines for training


class DataError(ValueError):
    pass


def validate_engine_id(engine_id, available_ids):
    try:
        eid = int(engine_id)
    except (TypeError, ValueError):
        raise DataError(f"Engine id {engine_id!r} is not a whole number.")
    if isinstance(engine_id, float) and engine_id != eid:
        raise DataError(f"Engine id {engine_id!r} is not a whole number.")
    if eid not in set(int(x) for x in available_ids):
        raise DataError(f"Engine {eid} does not exist in the loaded dataset.")
    return eid


def prepare_window(df, engine_id, sensors, scaler, window_size=WINDOW_SIZE):
    """
    Returns {"engine", "current_cycle", "recent" (raw DataFrame, window_size rows),
             "scaled" (float64 array, shape (window_size, n_sensors)), "warnings"}.
    """
    missing_cols = [c for c in ["unit", "cycle", *sensors] if c not in df.columns]
    if missing_cols:
        raise DataError("Missing required columns: " + ", ".join(missing_cols))

    eid = validate_engine_id(engine_id, df["unit"].unique())
    warnings = []

    engine = df[df["unit"] == eid].sort_values("cycle")
    if engine["cycle"].duplicated().any():
        engine = engine.drop_duplicates(subset="cycle", keep="last")
        warnings.append("Duplicate cycle rows were found; the last row of each cycle was kept.")

    engine = engine.copy()
    values = engine[sensors].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    all_missing = [s for s in sensors if values[s].isna().all()]
    if all_missing:
        raise DataError("No valid readings for: " + ", ".join(all_missing))

    n_bad = int(values.isna().sum().sum())
    if n_bad:
        values = values.ffill().bfill()
        warnings.append(f"{n_bad} missing or invalid sensor value(s) were filled from neighbouring cycles.")
    engine[sensors] = values

    current_cycle = int(engine["cycle"].max())
    recent = engine.tail(window_size)

    if len(recent) < window_size:
        pad = pd.concat([recent.iloc[[0]]] * (window_size - len(recent)), ignore_index=True)
        recent = pd.concat([pad, recent], ignore_index=True)
        warnings.append(
            f"Engine {eid} has only {len(engine)} cycles; the window was padded by repeating the "
            f"first row, so the prediction is less reliable."
        )

    scaled = np.asarray(scaler.transform(recent[sensors]), dtype=np.float64)
    if scaled.shape != (window_size, len(sensors)) or not np.all(np.isfinite(scaled)):
        raise DataError("Scaled input window is invalid.")

    return {"engine": engine, "current_cycle": current_cycle, "recent": recent,
            "scaled": scaled, "warnings": warnings}


# ------------------------------------------------------------
# Scaler consistency (the LSTM was trained on inputs scaled by preprocessing.py)
# ------------------------------------------------------------
def training_recipe_scaler(train_df, sensors, seed=SPLIT_SEED, train_fraction=TRAIN_FRACTION):
    """
    Re-creates the scaler exactly as preprocessing.py does: MinMaxScaler fitted on
    the 80% training ENGINES (seed 42 shuffle). Never uses validation or test data.
    """
    from sklearn.preprocessing import MinMaxScaler

    units = train_df["unit"].unique()
    np.random.seed(seed)
    np.random.shuffle(units)
    train_units = units[: int(len(units) * train_fraction)]
    scaler = MinMaxScaler()
    scaler.fit(train_df.loc[train_df["unit"].isin(train_units), sensors])
    return scaler


def compare_scalers(scaler, reference):
    """Returns (identical: bool, list of sensor positions whose min/max differ)."""
    diff = (~np.isclose(scaler.data_min_, reference.data_min_, rtol=0, atol=1e-9)) | \
           (~np.isclose(scaler.data_max_, reference.data_max_, rtol=0, atol=1e-9))
    return (not diff.any()), [int(i) for i in np.where(diff)[0]]
