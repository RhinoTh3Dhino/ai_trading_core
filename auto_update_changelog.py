import subprocess
import datetime
import os
import re

CHANGELOG_PATH = "CHANGELOG.md"


def get_last_version():
    if not os.path.exists(CHANGELOG_PATH):
        return "0.0.0"
    with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"## \[(\d+\.\d+\.\d+)\]", line)
            if match:
                return match.group(1)
    return "0.0.0"


def bump_version(version, level="patch"):
    major, minor, patch = [int(x) for x in version.split(".")]
    if level == "major":
        major += 1
        minor = patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def get_latest_commits(since_tag=None, max_count=10):
    args = ["git", "log", "--pretty=format:%h %an %ad %s", "--date=short"]
    if since_tag:
        args += [f"{since_tag}..HEAD"]
    output = subprocess.check_output(args, encoding="utf-8")
    return output.strip().split("\n")[:max_count]


def make_changelog_entry(version, commits):
    today = datetime.date.today().isoformat()
    entry = f"## [{version}] – {today}\n\n"
    entry += "**Seneste ændringer:**\n\n"
    for c in commits:
        entry += f"- {c}\n"
    entry += "\n---\n\n"
    return entry


def prepend_changelog(entry):
    if os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
            old = f.read()
    else:
        old = ""
    with open(CHANGELOG_PATH, "w", encoding="utf-8") as f:
        f.write(entry + old)


if __name__ == "__main__":
    # 1. Find sidste version i CHANGELOG.md
    last_version = get_last_version()
    # 2. Bump version (patch by default, men kan ændres)
    new_version = bump_version(last_version, level="minor")  # eller level="patch"
    # 3. Hent seneste commits (maks 10)
    commits = get_latest_commits()
    # 4. Generér changelog-entry
    entry = make_changelog_entry(new_version, commits)
    # 5. Skriv ind øverst i CHANGELOG.md
    prepend_changelog(entry)
    print(f"✅ CHANGELOG.md auto-opdateret (version {new_version})!")
