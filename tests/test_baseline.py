# tests/test_baseline.py

from utils.project_path import PROJECT_ROOT
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # ---- 1. Tjek target-fordeling ----
    target_path = PROJECT_ROOT / "data" / "BTCUSDT_1h_with_target.csv"
    if not target_path.exists():
        print(f"[FEJL] Mangler fil: {target_path}. Kør først generate_target.py.")
        sys.exit(1)
    
    df = pd.read_csv(target_path)
    print("\n=== Target fordeling ===")
    print(df['target'].value_counts(normalize=True))
    if df['target'].isna().any():
        print("[ADVARSEL] Der er NaN i target-kolonnen!")

    # ---- 2. Plot target over tid (valgfrit, gem billede) ----
    plt.figure(figsize=(8,3))
    df['target'].rolling(100).mean().plot(title="Target (TP-hit) over tid")
    plt.tight_layout()
    plot_path = PROJECT_ROOT / "outputs" / "target_rolling_mean.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Plot af target-rolling mean gemt som: {plot_path}")

    # ---- 3. Tjek test-resultater fra models ----
    result_path = PROJECT_ROOT / "outputs" / "feature_importance_baseline.png"
    if not result_path.exists():
        print(f"[ADVARSEL] Feature importance-billede ikke fundet. Har du kørt train_baseline.py?")
    else:
        print(f"[OK] Feature importance-billede findes: {result_path}")

    # ---- 4. Vis nøgletal fra sidste træning ----
    # Her kan du evt. parse/åbne model-history hvis du logger dine metrics til csv,
    # eller blot henvise til Telegram/terminal output.
    print("\n[INFO] Tjek terminal eller Telegram for winrate, accuracy og Sharpe fra sidste træning.")
    print("Krav for edge: Winrate > 55%, Sharpe > 0.8")
    print("Hvis ikke, så prøv at tune TP/SL eller ret targets!")

    # ---- 5. Ekstra konsistenskontrol ----
    print("\nFørste 3 rækker af data med target:")
    print(df.head(3))
    print("\nScriptet er færdigt. Se plot og evt. Telegram/terminal for resten af resultaterne.")

if __name__ == "__main__":
    main()
