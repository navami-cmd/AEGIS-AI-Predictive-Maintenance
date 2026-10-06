import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

files = [
    ('Baseline LSTM', 'test_predictions.csv'),
    ('Transformer', 'transformer_test_predictions.csv'),
]

rows = []
for name, filename in files:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    error = df['Predicted_RUL'] - df['Actual_RUL']
    rows.append({
        'Model': name,
        'MAE': np.mean(np.abs(error)),
        'RMSE': np.sqrt(np.mean(error ** 2)),
    })

if not rows:
    raise FileNotFoundError('No model prediction files found in data/processed.')

result = pd.DataFrame(rows)
print(result.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
result.to_csv(os.path.join(DATA_DIR, 'model_comparison.csv'), index=False)
print('\nSaved: data/processed/model_comparison.csv')
