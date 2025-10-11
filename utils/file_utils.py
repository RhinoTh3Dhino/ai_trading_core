import datetime
import os
import subprocess

import pandas as pd

from utils.project_path import PROJECT_ROOT

# utils/file_utils.py


def get_git_hash():
    """Hent aktiv git commit hash – eller 'unknown' hvis fejl."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception as e:
        print(f"[ADVARSEL] Kunne ikke hente git-hash: {e}")
        return "unknown"


def save_with_metadata(df, out_path, version="v1.0.0", extra_metadata=None):
    """
    Gem DataFrame til CSV med timestamp, git-hash, version og evt. ekstra metadata.
    Opretter mappe automatisk hvis den ikke findes.
    Tilføjer kun meta-info som kolonner hvis ikke allerede til stede.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()

    meta_cols = {"log_timestamp": timestamp, "git_hash": git_hash, "version": version}
    if extra_metadata and isinstance(extra_metadata, dict):
        meta_cols.update(extra_metadata)

    # Tilføj kun metadata-kolonner hvis de ikke allerede findes (for at undgå vokseværk)
    for col, val in meta_cols.items():
        if col not in df.columns:
            df[col] = val

    # Sikrer at output-mappen findes
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, index=False)
    # --- UNDGA EMOJI HER ---
    print(f"Fil gemt med metadata: {out_path} (v={version}, git={git_hash})")
    return out_path


# Eksempel/test
if __name__ == "__main__":
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # AUTO PATH CONVERTED
    save_with_metadata(
        test_df,
        PROJECT_ROOT / "outputs" / "test_out.csv",
        version="v1.2.3",
        extra_metadata={"source": "unittest"},
    )
