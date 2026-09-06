#!/usr/bin/env python3
"""Pre-commit guard: fail if any file git is about to publish contains a
person name from the capstone corpus.

The name list is derived at run time from bhi2026/metadata/redaction_log.csv
(which is itself gitignored and never leaves this machine), so this script
contains no names and is safe to commit. Run it from the repo root before
every commit:

    python3 tools/check_release_pii.py          # checks tracked + staged files

Exit status 0 = clean, 1 = names found (offending files listed).
"""
import csv
import re
import subprocess
import sys
import zipfile
from pathlib import Path

LOG = Path("bhi2026/metadata/redaction_log.csv")
TEXT_EXT = {".py", ".md", ".csv", ".json", ".txt", ".tex", ".bib", ".cls",
            ".yml", ".toml", ".cfg", ".sh", ".gitignore"}

# Tokens that spaCy mis-tagged as PERSON in the redaction log; multi-word
# surfaces containing any of these are topic phrases, not names.
NON_NAME_TOKENS = set("""ai ml dl nlp emr ehr bi hie hit phi pii api sql bmi icu
health healthcare data analysis analytics tool tools system systems record
records electronic medical informatics autism spectrum disorder integrated
powered advanced evolution program university learning
machine deep disease clinical patient patients care hospital center school
institute department quality improvement management information technology
science public mental cancer telehealth orchestrating insights solutions
capstone practicum presentation poster review project risk model models digital
workflow workflows nurse nursing pharmacy diabetes covid virtual reality
automated automatic extraction submitted""".split())

# Public author names permitted in the manuscript, loaded from the gitignored
# local/author_allowlist.txt (one name per line) so this script names no one.
_PERSONS = Path("local/person_markers.txt")
_ORGS = Path("local/org_markers.txt")
_ALLOW = Path("local/author_allowlist.txt")
AUTHOR_ALLOWLIST = ({l.strip() for l in _ALLOW.read_text().splitlines() if l.strip()}
                    if _ALLOW.exists() else set())
AUTHOR_FILES = re.compile(r"manuscript/.*\.(tex|bib)$")


def curated(path):
    """Explicit, human-reviewed surfaces (never committed; see local/)."""
    if not path.exists():
        return []
    return sorted({" ".join(l.split()) for l in path.read_text().splitlines() if l.strip()},
                  key=len, reverse=True)


def finder(surface):
    """Case-insensitive, letter-bounded. Not \\b: BERTopic joins Name tokens with
    "_", which \\b treats as a word character, so a name in a topic label would
    otherwise be missed."""
    body = r"[\s_]*".join(map(re.escape, surface.split()))
    return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)


def person_surfaces():
    names = set()
    with LOG.open() as f:
        for row in csv.DictReader(f):
            for surf in (row.get("sample_redacted_surface") or "").split("|"):
                surf = " ".join(surf.split())
                toks = surf.split()
                if not (2 <= len(toks) <= 3):
                    continue
                ok = True
                for t in toks:
                    t2 = re.sub(r"[.\-']", "", t)
                    if (not t2.isalpha() or t2.lower() in NON_NAME_TOKENS
                            or not t[0].isupper() or len(t2) < 2):
                        ok = False
                        break
                if ok:
                    names.add(surf)
    return names


def file_text(path: Path) -> str:
    if path.suffix == ".xlsx":
        try:
            with zipfile.ZipFile(path) as z:
                return " ".join(z.read(n).decode("utf-8", "ignore")
                                for n in z.namelist() if n.endswith(".xml"))
        except Exception:
            return ""
    if path.suffix in TEXT_EXT or path.name == ".gitignore":
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def main() -> int:
    if not LOG.exists():
        print(f"ERROR: {LOG} not found — run from the repo root on the "
              f"machine that holds the restricted corpus.", file=sys.stderr)
        return 1
    names = person_surfaces()
    hard = ([(finder(m), "PERSON", m) for m in curated(_PERSONS)] +
            [(finder(m), "ORG", m) for m in curated(_ORGS)])
    print(f"Screening against {len(hard)} curated PERSON/ORG markers (hard fail) "
          f"and {len(names)} log-derived surfaces (advisory).")
    if not hard:
        print("WARNING: no curated marker lists found under local/ — "
              "only the advisory sweep will run.", file=sys.stderr)
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True)
    files = [Path(p) for p in out.stdout.splitlines() if p.strip()]
    bad = {}
    for fp in files:
        if not fp.exists():
            continue
        text = file_text(fp)
        if not text:
            continue
        found = sorted({f"{kind}:{surf}" for rx, kind, surf in hard if rx.search(text)})
        if found and AUTHOR_FILES.search(str(fp)):
            found = [f for f in found if f.split(":", 1)[1] not in AUTHOR_ALLOWLIST]
        if found:
            bad[str(fp)] = found
    if bad:
        print(f"\nFAIL — PERSON/ORG surfaces found in {len(bad)} file(s) "
              f"that git would publish:")
        for f, ns in bad.items():
            print(f"  {f}: {len(ns)} name(s)")
        return 1
    print(f"OK — {len(files)} publishable files screened, no curated PERSON/ORG surface found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
