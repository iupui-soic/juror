"""Pull abstracts from 8 health-informatics journals via NCBI Entrez.

Two tiers, ISSN-anchored, no AI/ML keyword filter, 2022-01-01 to 2025-12-31.
Cap per-journal at 1500 abstracts (take the most recent if more are returned).

Outputs to bhi2026/journals/:
- per_journal/<key>.json   — list of articles with full metadata per journal
- journal_corpus.csv       — flat consolidated table (one row per article)
- manifest.json            — retrieval date, queries, raw and capped counts,
                              per-journal date range present in the result

Run: python3 bhi2026/fetch_journals.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from Bio import Entrez

OUT_DIR = Path("bhi2026/journals")
PER_J_DIR = OUT_DIR / "per_journal"
RAW_DIR = OUT_DIR / "per_journal_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_J_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Required by NCBI for rate-limit identification.
Entrez.email = os.environ.get("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.environ.get("ENTREZ_API_KEY")
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY
    REQUESTS_PER_SEC = 10
else:
    REQUESTS_PER_SEC = 3
SLEEP_BETWEEN_CALLS = 1.05 / REQUESTS_PER_SEC

# ISSN-anchored journal definitions. Where a journal has both print and online
# ISSNs we list both and OR them together. journals tested against PubMed
# 2026-05-21 and indexed under the listed ISSNs.
JOURNALS = [
    {
        "key": "JAMIA",
        "title": "Journal of the American Medical Informatics Association",
        "tier": "core",
        "issn": ["1067-5027", "1527-974X"],
    },
    {
        "key": "JBI",
        "title": "Journal of Biomedical Informatics",
        "tier": "core",
        "issn": ["1532-0464", "1532-0480"],
    },
    {
        "key": "JBHI",
        "title": "IEEE Journal of Biomedical and Health Informatics",
        "tier": "core",
        "issn": ["2168-2194", "2168-2208"],
    },
    {
        "key": "IJMI",
        "title": "International Journal of Medical Informatics",
        "tier": "core",
        "issn": ["1386-5056", "1872-8243"],
    },
    {
        "key": "ACI",
        "title": "Applied Clinical Informatics",
        "tier": "core",
        "issn": ["1869-0327"],
    },
    {
        "key": "JMIR",
        "title": "Journal of Medical Internet Research",
        "tier": "digital_health",
        "issn": ["1438-8871"],
    },
    {
        "key": "npjDigitalMed",
        "title": "npj Digital Medicine",
        "tier": "digital_health",
        "issn": ["2398-6352"],
    },
    {
        "key": "LancetDigitalHealth",
        "title": "The Lancet. Digital health",
        "tier": "digital_health",
        "issn": ["2589-7500"],
    },
]

DATE_FROM = "2020/01/01"
DATE_TO = "2025/12/31"
YEAR_RANGE = (2020, 2021, 2022, 2023, 2024, 2025)
CAP_PER_JOURNAL = 1500
ESEARCH_RETMAX = 10000  # PubMed allows up to 10k per esearch.
EFETCH_BATCH = 200
RNG_SEED = 42


def build_query(issns: list[str]) -> str:
    issn_clause = " OR ".join(f"{i}[IS]" for i in issns)
    return f"({issn_clause}) AND (\"{DATE_FROM}\"[PDAT] : \"{DATE_TO}\"[PDAT])"


def esearch_ids(query: str) -> list[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=ESEARCH_RETMAX,
        sort="pub date",
        usehistory="n",
    )
    record = Entrez.read(handle)
    handle.close()
    return list(record.get("IdList", []))


def fetch_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def parse_pubmed_articles(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        pmid = fetch_text(art.find(".//PMID"))
        title = fetch_text(art.find(".//ArticleTitle"))

        # Abstract: concatenate all AbstractText nodes, preserving section
        # labels when present (e.g. "BACKGROUND: ... METHODS: ...").
        abstract_parts = []
        for at in art.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            txt = fetch_text(at)
            if label and txt:
                abstract_parts.append(f"{label}: {txt}")
            elif txt:
                abstract_parts.append(txt)
        abstract = "\n".join(abstract_parts).strip()

        # Journal info
        journal_title = fetch_text(art.find(".//Journal/Title"))
        issn_print = fetch_text(art.find(".//Journal/ISSN[@IssnType='Print']"))
        issn_online = fetch_text(art.find(".//Journal/ISSN[@IssnType='Electronic']"))
        if not (issn_print or issn_online):
            # Any ISSN, regardless of type
            issn_print = fetch_text(art.find(".//Journal/ISSN"))

        # Publication date (Year)
        year = fetch_text(art.find(".//Journal/JournalIssue/PubDate/Year"))
        if not year:
            year = fetch_text(art.find(".//Journal/JournalIssue/PubDate/MedlineDate"))[:4]
        month = fetch_text(art.find(".//Journal/JournalIssue/PubDate/Month"))

        # DOI
        doi = ""
        for el in art.findall(".//ArticleId"):
            if el.get("IdType", "").lower() == "doi":
                doi = el.text or ""
                break

        # Authors (collapsed to last+initials, max 6)
        authors = []
        for au in art.findall(".//AuthorList/Author"):
            last = fetch_text(au.find("LastName"))
            init = fetch_text(au.find("Initials"))
            if last:
                authors.append(f"{last} {init}".strip())
            if len(authors) >= 6:
                break

        # MeSH headings (descriptor names)
        mesh = []
        for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            t = fetch_text(d)
            if t:
                mesh.append(t)

        # Article type(s) — useful for filtering editorials/letters later.
        ptypes = []
        for pt in art.findall(".//PublicationTypeList/PublicationType"):
            t = fetch_text(pt)
            if t:
                ptypes.append(t)

        articles.append({
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "journal_title": journal_title,
            "issn_print": issn_print,
            "issn_online": issn_online,
            "year": year,
            "month": month,
            "authors": authors,
            "mesh": mesh,
            "publication_types": ptypes,
        })
    return articles


def _fetch_xml(ids: list[str]) -> bytes:
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(ids),
        rettype="xml",
        retmode="xml",
    )
    data = handle.read()
    handle.close()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def efetch_articles(pmids: list[str]) -> list[dict]:
    """Fetch + parse PubMed records for a list of PMIDs, batched.

    On a batch-level XML parse error (PubMed occasionally returns malformed
    characters), fall back to fetching the offending batch one PMID at a time
    so we lose only the actual bad article, not the whole batch.
    """
    out: list[dict] = []
    for i in range(0, len(pmids), EFETCH_BATCH):
        batch = pmids[i : i + EFETCH_BATCH]
        time.sleep(SLEEP_BETWEEN_CALLS)
        try:
            xml_bytes = _fetch_xml(batch)
            parsed = parse_pubmed_articles(xml_bytes)
        except ET.ParseError as e:
            print(f"    XML parse error on batch (size {len(batch)}): {e}; falling back to per-article", flush=True)
            parsed = []
            dropped = 0
            for pmid in batch:
                time.sleep(SLEEP_BETWEEN_CALLS)
                try:
                    one_xml = _fetch_xml([pmid])
                    parsed.extend(parse_pubmed_articles(one_xml))
                except (ET.ParseError, Exception) as inner:
                    print(f"      dropped PMID {pmid}: {type(inner).__name__}", flush=True)
                    dropped += 1
            if dropped:
                print(f"    recovered {len(parsed)} / {len(batch)} (dropped {dropped})", flush=True)
        out.extend(parsed)
        print(f"    fetched batch {i//EFETCH_BATCH + 1}: {len(parsed)} articles (cumulative {len(out)})", flush=True)
    return out


def cap_year_stratified(articles: list[dict], cap: int) -> list[dict]:
    """Year-stratified cap.

    Aim for cap/len(YEAR_RANGE) articles per year (375 each for cap=1500 and a
    4-year window). If a year is short, redistribute the leftover quota to
    years with surplus. Within a year sample randomly with a fixed seed.
    """
    import random
    rnd = random.Random(RNG_SEED)

    by_year: dict[int, list[dict]] = {y: [] for y in YEAR_RANGE}
    for a in articles:
        try:
            y = int(a["year"])
        except (ValueError, TypeError):
            continue
        if y in by_year:
            by_year[y].append(a)

    quota = cap // len(YEAR_RANGE)
    remaining = cap - quota * len(YEAR_RANGE)  # rounding leftover

    picked: list[dict] = []
    shortfalls: list[int] = []
    surplus: list[int] = []
    for y in YEAR_RANGE:
        pool = by_year[y]
        if len(pool) <= quota:
            picked.extend(pool)
            if len(pool) < quota:
                shortfalls.append(quota - len(pool))
                surplus.append(0)
            else:
                shortfalls.append(0)
                surplus.append(0)
        else:
            sample = rnd.sample(pool, quota)
            picked.extend(sample)
            shortfalls.append(0)
            surplus.append(len(pool) - quota)

    # Redistribute shortfalls: pick from years that have surplus, in order of
    # surplus size.
    leftover = sum(shortfalls) + remaining
    if leftover > 0:
        for idx in sorted(range(len(YEAR_RANGE)), key=lambda i: -surplus[i]):
            if leftover <= 0:
                break
            y = YEAR_RANGE[idx]
            already = quota if surplus[idx] > 0 else len(by_year[y])
            available_pool = [a for a in by_year[y] if a not in picked]
            take = min(leftover, len(available_pool))
            if take > 0:
                extras = rnd.sample(available_pool, take)
                picked.extend(extras)
                leftover -= take

    return picked


def filter_to_window(articles: list[dict]) -> list[dict]:
    """Strict year filter — drop publications outside YEAR_RANGE.

    PubMed sometimes returns articles with PubDate in 2026 because of
    accelerated-online publication; we want them out for clean 2022-2025 spans.
    """
    out = []
    for a in articles:
        try:
            y = int(a["year"])
        except (ValueError, TypeError):
            continue
        if y in YEAR_RANGE:
            out.append(a)
    return out


def main() -> None:
    manifest: dict = {
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": [DATE_FROM, DATE_TO],
        "cap_per_journal": CAP_PER_JOURNAL,
        "journals": [],
    }

    all_articles: list[dict] = []

    for j in JOURNALS:
        query = build_query(j["issn"])
        print(f"\n=== {j['key']} ({j['title']}) ===", flush=True)
        print(f"  query: {query}", flush=True)

        raw_path = RAW_DIR / f"{j['key']}.json"
        if raw_path.exists():
            print(f"  raw cache hit: loading {raw_path}", flush=True)
            articles = json.loads(raw_path.read_text())
            n_raw = len(articles)
            n_with_content = sum(1 for a in articles if a["title"] or a["abstract"])
            articles = [a for a in articles if (a["title"] or a["abstract"])]
            for a in articles:
                a.setdefault("journal_key", j["key"])
                a.setdefault("tier", j["tier"])
        else:
            time.sleep(SLEEP_BETWEEN_CALLS)
            pmids = esearch_ids(query)
            n_raw = len(pmids)
            print(f"  esearch returned {n_raw} PMIDs", flush=True)

            articles = efetch_articles(pmids)
            articles = [a for a in articles if (a["title"] or a["abstract"])]
            n_with_content = len(articles)

            for a in articles:
                a["journal_key"] = j["key"]
                a["tier"] = j["tier"]
            raw_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

        in_window = filter_to_window(articles)
        n_in_window = len(in_window)
        capped = cap_year_stratified(in_window, CAP_PER_JOURNAL)

        # Per-journal JSON
        per_path = PER_J_DIR / f"{j['key']}.json"
        per_path.write_text(json.dumps(capped, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> kept {len(capped)} (in-window {n_in_window}, content {n_with_content}, raw {n_raw}); wrote {per_path}", flush=True)

        all_articles.extend(capped)

        # Year breakdown in retained set
        from collections import Counter
        year_breakdown = Counter(a["year"] for a in capped if a["year"].isdigit())
        manifest["journals"].append({
            "key": j["key"],
            "title": j["title"],
            "tier": j["tier"],
            "issn": j["issn"],
            "query": query,
            "n_raw": n_raw,
            "n_with_content": n_with_content,
            "n_in_window": n_in_window,
            "n_kept": len(capped),
            "year_breakdown": dict(sorted(year_breakdown.items())),
        })

    # Consolidated CSV (flatten authors/mesh)
    csv_path = OUT_DIR / "journal_corpus.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "journal_key", "tier", "pmid", "doi", "year", "month",
            "journal_title", "issn_print", "issn_online",
            "title", "abstract", "authors", "mesh", "publication_types",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in all_articles:
            row = {k: a.get(k, "") for k in fields if k not in {"authors", "mesh", "publication_types"}}
            row["authors"] = "|".join(a.get("authors", []))
            row["mesh"] = "|".join(a.get("mesh", []))
            row["publication_types"] = "|".join(a.get("publication_types", []))
            writer.writerow(row)
    print(f"\nWrote consolidated CSV: {csv_path} ({len(all_articles)} rows)", flush=True)

    # Manifest
    manifest["total_kept"] = len(all_articles)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary
    print("\n=== Per-journal summary ===", flush=True)
    print(f"{'Journal':<22} {'tier':<14} {'raw':>6} {'in_win':>6} {'kept':>6}  by year", flush=True)
    for m in manifest["journals"]:
        yr_str = " ".join(f"{y}:{c}" for y, c in m["year_breakdown"].items())
        print(f"{m['key']:<22} {m['tier']:<14} {m['n_raw']:>6} {m['n_in_window']:>6} {m['n_kept']:>6}  {yr_str}", flush=True)
    print(f"\nTotal articles in corpus: {len(all_articles)}", flush=True)


if __name__ == "__main__":
    main()
