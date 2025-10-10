import os
import shutil
from datetime import datetime

INCLUDE = [
    ("outputs/models", True),
    ("outputs/backtests", True),
    ("outputs/metrics", True),
    ("outputs/charts", True),
    ("outputs/feature_data", False),
    ("outputs/labels", False),
    ("docs", True),
    ("BotStatus.md", False),
    ("CHANGELOG.md", False),
]


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = "archives"
    os.makedirs(outdir, exist_ok=True)
    root = f"ai_trading_core_{stamp}"
    tmpdir = os.path.join(outdir, root)
    os.makedirs(tmpdir, exist_ok=True)
    for path, required in INCLUDE:
        if not os.path.exists(path):
            if required:
                print(f"⚠ mangler: {path}")
            continue
        dest = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isdir(path):
            shutil.copytree(path, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(path, dest)
    zip_path = shutil.make_archive(os.path.join(outdir, root), "zip", tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"✅ Arkiv: {zip_path}")


if __name__ == "__main__":
    main()
