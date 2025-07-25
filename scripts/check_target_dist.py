import pandas as pd

# Indlæs dine feature-data (efter generate_features.py)
df = pd.read_csv("data/BTCUSDT_1h_features.csv")

# Loop igennem alle kolonner, der starter med "target_"
for col in [c for c in df.columns if c.startswith("target_")]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False))  # viser antal 1, 0 og evt. NaN
    pct_1 = (df[col] == 1).mean() * 100
    pct_0 = (df[col] == 0).mean() * 100
    print(f"Procent 1: {pct_1:.2f}%, Procent 0: {pct_0:.2f}%")
