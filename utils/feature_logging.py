import datetime
import os

import pandas as pd

from utils.project_path import PROJECT_ROOT


def log_top_features_to_md(
    top_features, md_path="BotStatus.md", model_name="ML", run_time=None
):
    """Logger top-5 features til BotStatus.md med timestamp og modelnavn."""
    if run_time is None:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n### Top-5 features ({model_name}) – {run_time}\n"
    lines = [
        f"{i+1}. {name}: {score:.4f}" for i, (name, score) in enumerate(top_features)
    ]
    entry = header + "\n".join(lines) + "\n"
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"✅ Top-5 features logget til {md_path}")


# AUTO PATH CONVERTED
def log_top_features_csv(
    top_features,
    csv_path=PROJECT_ROOT / "data" / "top_features_history.csv",
    model_name="ML",
    run_time=None,
):
    """Gemmer top-5 features med importance i CSV for historisk sammenligning."""
    if run_time is None:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(top_features, columns=["feature", "importance"])
    df["model"] = model_name
    df["timestamp"] = run_time
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"✅ Top-5 features logget til {csv_path}")


def send_top_features_telegram(
    top_features, send_telegram_message, chat_id, model_name="ML"
):
    msg = f"📊 Top-5 features ({model_name}):\n" + "\n".join(
        [f"{i+1}. {name}: {score:.4f}" for i, (name, score) in enumerate(top_features)]
    )
    send_telegram_message(msg, chat_id=chat_id)
