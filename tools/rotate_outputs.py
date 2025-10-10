# tools/rotate_outputs.py


from utils.artifacts import rotate_dir

TARGETS = {
    "outputs/feature_data": (3, r".*\.(csv|parquet)$"),  # små tal for nem test
    "outputs/labels": (3, r".*\.npy$"),
    "outputs/models": (2, r".*\.(keras|h5)$"),
    "outputs/backtests": (3, r".*\.(csv|png)$"),
    "outputs/metrics": (5, r".*\.json$"),
    "outputs/charts": (5, r".*\.png$"),
    "outputs/logs": (5, r".*\.log$"),
    "archives": (5, r".*\.zip$"),
}
if __name__ == "__main__":
    for d, (keep, pat) in TARGETS.items():
        rotate_dir(d, keep=keep, pattern=pat)
    print("✅ Rotation complete")
