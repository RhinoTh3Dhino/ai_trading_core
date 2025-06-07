import datetime
import subprocess

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        return "unknown"

def save_with_metadata(df, out_path, version="v1.0.0"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = get_git_hash()
    df["log_timestamp"] = timestamp
    df["git_hash"] = git_hash
    df["version"] = version
    df.to_csv(out_path, index=False)
