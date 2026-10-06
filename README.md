# AEGIS AI — Final Prototype Model Package

This package preserves the existing working Streamlit dashboard and adds the remaining model-development scripts.

## Existing baseline
- C-MAPSS FD001
- 17 useful sensors
- 30-cycle windows
- 2-layer LSTM, 64 hidden units, dropout 0.2
- FC: 64 -> 32 -> 1
- Test MAE: 16.66 cycles
- Test RMSE: 23.60 cycles

## Dashboard fix
`app.py` keeps the existing AEGIS dashboard and uses `test_predictions.csv` for the primary displayed engine RUL when that file exists. MC Dropout remains responsible for uncertainty. This makes engine selection agree with the saved evaluation results instead of replacing the evaluated prediction with a different stochastic mean.

## New model-development scripts
### 1. Piecewise RUL LSTM
```powershell
python train_piecewise_lstm.py
```
Creates:
- `data/processed/piecewise_lstm_rul_model.pth`

The RUL target is capped at 125 cycles during training. This is an experimental improved model; it does not replace the baseline automatically.

### 2. Transformer
```powershell
python train_transformer.py
```
Creates:
- `data/processed/transformer_rul_model.pth`
- `data/processed/transformer_test_predictions.csv`

### 3. Compare models
Run this after Transformer training:
```powershell
python compare_models.py
```
Creates:
- `data/processed/model_comparison.csv`

## Dashboard
Keep the existing project data/model folders beside `app.py`, then run:
```powershell
python -m streamlit run app.py
```

The existing `agents.py` is intentionally not changed.
