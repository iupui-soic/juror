#!/usr/bin/env python3
"""Apply the curated PERSON/ORG markers to files git would publish.

Companion to check_release_pii.py: that script detects, this one repairs. Both
read the same gitignored marker lists (local/person_markers.txt,
local/org_markers.txt), so neither names anyone.

Needed because a name can reach a release file without ever appearing in a
document: BERTopic's c-TF-IDF promotes a frequent token to a topic keyword and
lower-cases it, the topic Name embeds it as "4_spss_depression_statistical_<name>",
and every downstream artifact that quotes topic labels inherits it.

    python3 tools/scrub_release_pii.py --dry-run     # show what would change
    python3 tools/scrub_release_pii.py               # rewrite in place
"""
import re, subprocess, sys
from pathlib import Path

PERSONS = Path("local/person_markers.txt")
ORGS = Path("local/org_markers.txt")
EXT = {".csv", ".md", ".json", ".txt", ".tex"}
SKIP = {"tools/scrub_release_pii.py", "tools/check_release_pii.py"}


def patterns():
    out = []
    for path, tag in ((ORGS, "[ORG]"), (PERSONS, "[PERSON]")):
        if not path.exists():
            continue
        seen = {" ".join(l.split()) for l in path.read_text().splitlines() if l.strip()}
        for m in sorted(seen, key=len, reverse=True):
            body = r"[\s_]*".join(map(re.escape, m.split()))
            out.append((re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE), tag))
    return out


def main():
    dry = "--dry-run" in sys.argv
    pats = patterns()
    if not pats:
        print("ERROR: no marker lists under local/", file=sys.stderr); return 1
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, text=True, check=True).stdout.split()
    total = 0
    for f in files:
        p = Path(f)
        if f in SKIP or not p.exists() or p.suffix not in EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = text, 0
        for rx, tag in pats:
            new, k = rx.subn(tag, new)
            n += k
        if n:
            total += n
            print(f"  {f}: {n} substitution(s){' (dry run)' if dry else ''}")
            if not dry:
                p.write_text(new, encoding="utf-8")
    print(f"{'Would rewrite' if dry else 'Rewrote'} {total} substitution(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
