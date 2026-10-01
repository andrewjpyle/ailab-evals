"""Privacy-wall check: fail if any tracked file contains a denied term, a private-network address,
a secrets-manager token, or a non-noreply email address.

Denied terms are stored as sha256 hashes in .denylist.sha256 so the list itself is safe to publish.
Every word token AND every dotted host-like token (e.g. ``a.b.c``) in each file is hashed and
looked up. A plaintext extra list can be supplied via AILAB_DENYLIST_FILE (kept outside the repo).

Exit 0 = clean, 1 = violation. Runs in the pre-push hook and as a required CI job.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"[a-z0-9]+")
HOSTLIKE = re.compile(r"[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+")
PATTERNS = {
    "tailnet/CGNAT address": re.compile(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "tailnet hostname": re.compile(r"\b[\w-]+\.ts\.net\b"),
    "secrets-manager token": re.compile(r"\bdp\.(st|pt|ct|sa|scim|audit)\.(?:[\w-]+\.)?[A-Za-z0-9]{20,}"),
    "private email": re.compile(r"[\w.+-]+@(?!users\.noreply\.github\.com|example\.(?:com|org))[\w-]+\.[\w.-]+"),
}
SKIP = {".denylist.sha256"}


def load_hashes() -> set[str]:
    hashes = {
        line.strip() for line in (ROOT / ".denylist.sha256").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    extra = os.environ.get("AILAB_DENYLIST_FILE")
    if extra and Path(extra).exists():
        for line in Path(extra).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                hashes.add(hashlib.sha256(line.strip().lower().encode()).hexdigest())
    return hashes


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return [ROOT / f for f in out.splitlines() if f and Path(f).name not in SKIP]


def scan(text: str, hashes: set[str]) -> list[str]:
    low = text.lower()
    hits = []
    for tok in set(WORD.findall(low)) | set(HOSTLIKE.findall(low)):
        if hashlib.sha256(tok.encode()).hexdigest() in hashes:
            hits.append(f"denied term (sha256 {hashlib.sha256(tok.encode()).hexdigest()[:12]})")
    for label, pat in PATTERNS.items():
        if pat.search(text):
            hits.append(label)
    return hits


def main() -> int:
    hashes = load_hashes()
    bad = 0
    for path in tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for hit in scan(text, hashes):
            bad += 1
            print(f"DENY {path.relative_to(ROOT)}: {hit}")
    print("denylist: clean" if not bad else f"denylist: {bad} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
