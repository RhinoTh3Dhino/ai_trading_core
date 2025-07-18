# auto_clean_models.py

import os

MODEL_DIR = "models"

# Hvilke filer skal BEHOLDES (ML og XGBoost/LightGBM, kan tilpasses)
KEEP_FILES = [
    "best_ml_model.pkl",
    "best_ml_features.json",
    "best_xgboost_model.json",
    "xgboost_features.json",
    "best_lightgbm_model.txt",
    "lightgbm_features.json",
]

def auto_clean_models(model_dir=MODEL_DIR, keep_files=KEEP_FILES, dry_run=False):
    """Sletter alle filer i model_dir undtagen dem i keep_files."""
    deleted = []
    kept = []
    for fname in os.listdir(model_dir):
        path = os.path.join(model_dir, fname)
        if fname not in keep_files and os.path.isfile(path):
            print(f"🗑️ Sletter: {fname}")
            if not dry_run:
                os.remove(path)
            deleted.append(fname)
        else:
            kept.append(fname)
    print(f"\n✅ Beholder: {', '.join(kept)}")
    print(f"🗑️ Slettet: {', '.join(deleted)}")
    return deleted, kept

if __name__ == "__main__":
    auto_clean_models()
