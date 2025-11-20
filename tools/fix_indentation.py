# tools/fix_indentation.py
import pathlib

ROOT = pathlib.Path(".")
for p in ROOT.rglob("*.py"):
    s = p.read_text(encoding="utf-8", errors="replace")
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = s.replace("\t", "    ")  # tab -> 4 spaces
    s = s.replace("\r\n", "\n")  # CRLF -> LF (valgfrit)
    p.write_text(s, encoding="utf-8")
print("Whitespace normaliseret.")
