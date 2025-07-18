import pandas as pd

df = pd.read_csv("data/BTCUSDT_1h_features.csv")
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 hvis næste close > nuværende close
df = df.dropna()  # Fjerner sidste række hvor der ikke er en "næste" close
df.to_csv("data/BTCUSDT_1h_features.csv", index=False)
print("✅ Target-kolonne tilføjet og gemt.")
