import pandas as pd

from utils.project_path import PROJECT_ROOT
# AUTO PATH CONVERTED
df = pd.read_csv(PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv")
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 hvis næste close > nuværende close
df = df.dropna()  # Fjerner sidste række hvor der ikke er en "næste" close
# AUTO PATH CONVERTED
df.to_csv(PROJECT_ROOT / "data" / "BTCUSDT_1h_features.csv", index=False)
print("✅ Target-kolonne tilføjet og gemt.")