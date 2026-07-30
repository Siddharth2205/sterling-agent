"""SEC EDGAR fundamentals — free, point-in-time, survivorship-free.

The Sharadar plan covers prices only, so fundamentals come straight from the source:
every US filer's XBRL facts via SEC EDGAR. Verified properties (2026-07-30):

  - Bulk file: https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
    (~1.4 GB, one JSON per filer, ALL filers incl. delisted — survivorship-free).
  - Every fact carries its `filed` date → `available_from` = filing date, no look-ahead.
  - Requests MUST send a User-Agent header or the SEC returns 403.
  - XBRL history starts ~2009 (mandate phase-in); earlier rows simply have no
    fundamentals, which the model handles as missing.

Tags are unstandardized (each company picks its us-gaap tag), so `TAG_MAP` holds a
priority list per metric; the first tag with enough facts wins per company. TTM flows
are built from discrete quarters, with Q4 derived from the annual filing (FY − Q1..Q3).
A record's `available_from` is the LATEST filed date among every fact used in it — the
conservative point-in-time rule.

Output records feed `fundamentals.fundamentals_as_of` unchanged:
  {available_from, revenue_ttm, revenue_ttm_prior, net_income_ttm,
   total_debt, total_equity, shares, fcf_ttm}
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sterling.research import config

logger = logging.getLogger(__name__)

ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
EDGAR_DIR = config.DATA / "edgar"
ZIP_PATH = EDGAR_DIR / "companyfacts.zip"
FACTS_PARQUET = EDGAR_DIR / "edgar_facts.parquet"
USER_AGENT = "Sterling Research sidinregina@gmail.com"

# Priority-ordered us-gaap tags per metric (first tag with enough facts wins).
# "flow" metrics are durations (quarter/year spans); "instant" are balance-sheet points.
TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "debt": [
        "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
    ],
}
FLOW_METRICS = ("revenue", "net_income", "ocf", "capex")
INSTANT_METRICS = ("equity", "debt")
# Share count lives in the `dei` namespace (unit "shares"), with a us-gaap fallback.
SHARES_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
               ("us-gaap", "CommonStockSharesOutstanding")]

_ALL_GAAP_TAGS = {t for tags in TAG_MAP.values() for t in tags}


# ── download + convert ──────────────────────────────────────────────────────────

def download_companyfacts(force: bool = False) -> Path:
    """Download the all-filers bulk zip (~1.4 GB). One-time; idempotent."""
    import requests
    if ZIP_PATH.exists() and not force:
        return ZIP_PATH
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("downloading SEC companyfacts.zip (~1.4 GB)...")
    with requests.get(ZIP_URL, headers={"User-Agent": USER_AGENT},
                      stream=True, timeout=3600) as r:
        r.raise_for_status()
        tmp = ZIP_PATH.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(ZIP_PATH)
    logger.info(f"  wrote {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")
    return ZIP_PATH


def _facts_from_json(obj: dict) -> list[dict]:
    """Extract the tag-mapped facts from one filer's companyfacts JSON."""
    cik = int(obj.get("cik", 0))
    rows: list[dict] = []
    gaap = obj.get("facts", {}).get("us-gaap", {})
    for metric, tags in TAG_MAP.items():
        for tag in tags:
            node = gaap.get(tag)
            if not node:
                continue
            for fact in node.get("units", {}).get("USD", []):
                if fact.get("val") is None or not fact.get("end") or not fact.get("filed"):
                    continue
                rows.append({
                    "cik": cik, "metric": metric, "tag": tag,
                    "start": fact.get("start"), "end": fact["end"],
                    "val": float(fact["val"]), "filed": fact["filed"],
                })
    for ns, tag in SHARES_TAGS:
        node = obj.get("facts", {}).get(ns, {}).get(tag)
        if not node:
            continue
        for fact in node.get("units", {}).get("shares", []):
            if fact.get("val") is None or not fact.get("end") or not fact.get("filed"):
                continue
            rows.append({
                "cik": cik, "metric": "shares", "tag": tag,
                "start": None, "end": fact["end"],
                "val": float(fact["val"]), "filed": fact["filed"],
            })
    return rows


def convert_companyfacts(zip_path: Optional[Path] = None,
                         parquet_path: Optional[Path] = None,
                         ciks: Optional[set[int]] = None,
                         force: bool = False) -> Path:
    """Stream the bulk zip → one slim Parquet of tag-mapped facts.

    Zip members are parsed in memory one at a time (never extracted to disk). Pass
    `ciks` to parse only the universe's filers — roughly halves the work.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    zip_path = zip_path or ZIP_PATH
    parquet_path = parquet_path or FACTS_PARQUET
    if parquet_path.exists() and not force:
        return parquet_path
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found — run download_companyfacts() first")

    schema = pa.schema([
        ("cik", pa.int64()), ("metric", pa.string()), ("tag", pa.string()),
        ("start", pa.string()), ("end", pa.string()),
        ("val", pa.float64()), ("filed", pa.string()),
    ])
    name_re = re.compile(r"CIK(\d{10})\.json$")
    parsed = skipped = 0
    buf: list[dict] = []
    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                m = name_re.search(name)
                if not m:
                    continue
                if ciks is not None and int(m.group(1)) not in ciks:
                    skipped += 1
                    continue
                try:
                    obj = json.loads(z.read(name))
                except Exception:  # noqa: BLE001 — one corrupt filer must not kill the run
                    continue
                buf.extend(_facts_from_json(obj))
                parsed += 1
                if len(buf) >= 200_000:
                    writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                    buf = []
                if parsed % 2000 == 0:
                    logger.info(f"  parsed {parsed} filers...")
            if buf:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
    finally:
        writer.close()
    logger.info(f"edgar facts: parsed {parsed} filers (skipped {skipped}) -> "
                f"{parquet_path.name} ({parquet_path.stat().st_size / 1e6:.0f} MB)")
    return parquet_path


# ── ticker ↔ CIK (incl. delisted, via the Sharadar tickers table) ───────────────

def ticker_cik_map() -> dict[str, int]:
    """{ticker: cik} from the Sharadar tickers table's secfilings URL — covers
    delisted names too, which the SEC's own ticker map does not."""
    from sterling.research import store
    meta = store.load_tickers()
    if "secfilings" not in meta.columns:
        return {}
    out: dict[str, int] = {}
    pat = re.compile(r"CIK=(\d+)")
    for tk, url in zip(meta["ticker"], meta["secfilings"].fillna("").astype(str)):
        m = pat.search(url)
        if m and pd.notna(tk):
            out[str(tk)] = int(m.group(1))
    return out


# ── point-in-time record construction ───────────────────────────────────────────

def _ranked(facts: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Facts under any priority tag, with a `rank` column (0 = most preferred).

    Companies switch tags over the years (e.g. SalesRevenueNet → RevenueFromContract...
    after the 2018 revenue standard), so eras must be MERGED per period, not one tag
    picked for the whole history."""
    sub = facts[facts["tag"].isin(tags)].copy()
    sub["rank"] = sub["tag"].map({t: i for i, t in enumerate(tags)})
    return sub


def _dedupe(facts: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One fact per period: prefer the higher-priority tag, then the EARLIEST filed
    (amendments must not rewrite what was knowable at the time)."""
    if "rank" not in facts.columns:
        facts = facts.assign(rank=0)
    return (facts.sort_values(["rank", "filed"])
            .drop_duplicates(subset=keys, keep="first"))


def _quarterly_series(facts: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarterly flows [end, val, filed] from whatever the filer reported.

    Three sources, in order:
      1. discrete ~3-month facts (income statements usually have them);
      2. differences of same-fiscal-year-start cumulatives — 10-Q cash-flow statements
         are filed as 6/9/12-month year-to-date, so Q_n = YTD_n − YTD_{n-1};
      3. Q4 = annual − (Q1+Q2+Q3) as a final fallback.
    A derived quarter is knowable only once BOTH its inputs are filed → filed = max.
    """
    f = facts.dropna(subset=["start"]).copy()
    if f.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    f["days"] = (f["end"] - f["start"]).dt.days
    f = _dedupe(f[(f["days"] >= 70) & (f["days"] <= 400)], ["start", "end"])

    q = f[f["days"] <= 105]
    rows = [{"end": r.end, "val": r.val, "filed": r.filed} for r in q.itertuples(index=False)]
    have = {r["end"] for r in rows}

    # cumulative YTD differencing (same fiscal-year start, ends one quarter apart)
    for _, g in f.groupby("start"):
        g = g.sort_values("days")
        for i in range(1, len(g)):
            prev, cur = g.iloc[i - 1], g.iloc[i]
            gap = (cur["end"] - prev["end"]).days
            if 70 <= gap <= 105 and cur["end"] not in have:
                rows.append({"end": cur["end"], "val": cur["val"] - prev["val"],
                             "filed": max(cur["filed"], prev["filed"])})
                have.add(cur["end"])

    # annual − three discrete quarters inside it
    a = f[f["days"] >= 330]
    for r in a.itertuples(index=False):
        inside = q[(q["start"] >= r.start) & (q["end"] <= r.end)]
        if len(inside) == 3 and r.end not in have:
            rows.append({"end": r.end, "val": r.val - inside["val"].sum(),
                         "filed": max(r.filed, inside["filed"].max())})
            have.add(r.end)

    if not rows:
        return pd.DataFrame(columns=["end", "val", "filed"])
    return pd.DataFrame(rows).sort_values("end").reset_index(drop=True)


def _ttm(qs: pd.DataFrame, i: int) -> Optional[tuple[float, pd.Timestamp]]:
    """(sum, max filed) of quarters i-3..i if they chain into a real trailing year."""
    if i < 3:
        return None
    w = qs.iloc[i - 3:i + 1]
    span = (w["end"].iloc[-1] - w["end"].iloc[0]).days
    if not (240 <= span <= 300):        # 3 gaps of ~91d — rejects series with holes
        return None
    return float(w["val"].sum()), w["filed"].max()


def records_for_cik(facts: pd.DataFrame) -> list[dict]:
    """Point-in-time records for one filer, ascending by available_from.

    Each record is stamped with the latest filed date among every fact it uses, so
    `fundamentals_as_of` can never see a number before the market did.
    """
    if facts.empty:
        return []
    facts = facts.copy()
    for c in ("start", "end", "filed"):
        facts[c] = pd.to_datetime(facts[c], errors="coerce")

    flows: dict[str, pd.DataFrame] = {}
    for m in FLOW_METRICS:
        flows[m] = _quarterly_series(_ranked(facts[facts["metric"] == m], TAG_MAP[m]))
    instants: dict[str, pd.DataFrame] = {}
    for m in INSTANT_METRICS:
        instants[m] = _dedupe(_ranked(facts[facts["metric"] == m], TAG_MAP[m]),
                              ["end"]).sort_values("end")
    instants["shares"] = _dedupe(
        _ranked(facts[facts["metric"] == "shares"], [t for _, t in SHARES_TAGS]),
        ["end"]).sort_values("end")

    # Anchor the record timeline on net income quarters (present for ~every filer),
    # falling back to revenue.
    anchor = flows["net_income"] if len(flows["net_income"]) else flows["revenue"]
    if anchor.empty:
        return []

    def instant_asof(m: str, qend: pd.Timestamp):
        s = instants[m]
        s = s[s["end"] <= qend + timedelta(days=5)]
        if s.empty:
            return None, None
        r = s.iloc[-1]
        return float(r["val"]), r["filed"]

    records = []
    rev = flows["revenue"]
    for i in range(len(anchor)):
        qend = anchor["end"].iloc[i]
        filed_dates = []

        def take_ttm(m: str):
            qs = flows[m]
            j = qs.index[qs["end"] == qend]
            if len(j) == 0:
                return None
            got = _ttm(qs, int(j[0]))
            if got is None:
                return None
            filed_dates.append(got[1])
            return got[0]

        ni = take_ttm("net_income")
        revenue = take_ttm("revenue")
        ocf = take_ttm("ocf")
        capex = take_ttm("capex")
        # No capex facts at all (banks/insurers) → FCF ≈ OCF; capex merely missing
        # this quarter → unknown, don't fake it.
        if ocf is None:
            fcf = None
        elif capex is None:
            fcf = ocf if flows["capex"].empty else None
        else:
            fcf = ocf - capex

        rev_prior = None
        j = rev.index[rev["end"] == qend]
        if len(j) and int(j[0]) >= 7:
            got = _ttm(rev, int(j[0]) - 4)
            if got is not None:
                rev_prior = got[0]     # already public by the current TTM's filing

        vals = {}
        for m, key in (("equity", "total_equity"), ("debt", "total_debt"), ("shares", "shares")):
            v, fd = instant_asof(m, qend)
            vals[key] = v
            if fd is not None:
                filed_dates.append(fd)

        if ni is None and revenue is None and vals["total_equity"] is None:
            continue
        records.append({
            "available_from": max(filed_dates).date().isoformat(),
            "revenue_ttm": revenue, "revenue_ttm_prior": rev_prior,
            "net_income_ttm": ni, "fcf_ttm": fcf,
            "total_equity": vals["total_equity"], "total_debt": vals["total_debt"],
            "shares": vals["shares"],
        })
    records.sort(key=lambda r: r["available_from"])
    return records


# ── pipeline glue ───────────────────────────────────────────────────────────────

def load_facts(ciks: Iterable[int], parquet: Optional[Path] = None) -> pd.DataFrame:
    """All extracted facts for `ciks` from the Parquet, via DuckDB."""
    import duckdb
    parquet = parquet or FACTS_PARQUET
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found — run `python -m sterling.research edgar`")
    con = duckdb.connect()
    try:
        con.register("want", pd.DataFrame({"cik": list(set(ciks))}))
        return con.execute(
            f"SELECT f.* FROM read_parquet('{parquet.as_posix()}') f "
            f"JOIN want USING (cik)"
        ).df()
    finally:
        con.close()


def fundamentals_fn(tickers: Optional[Iterable[str]] = None):
    """Build `hist_fund_fn(ticker) -> records` for features.build_features.

    Loads and processes facts for the whole universe up front (one Parquet scan),
    then serves per-ticker records from memory.
    """
    cik_of = ticker_cik_map()
    if tickers is not None:
        cik_of = {t: c for t, c in cik_of.items() if t in set(tickers)}
    facts = load_facts(cik_of.values())
    by_cik = dict(tuple(facts.groupby("cik")))
    cache: dict[str, list] = {}

    def fn(ticker: str) -> list[dict]:
        if ticker in cache:
            return cache[ticker]
        cik = cik_of.get(ticker)
        recs = records_for_cik(by_cik[cik]) if cik in by_cik else []
        cache[ticker] = recs
        return recs

    logger.info(f"edgar fundamentals ready: {len(by_cik)} filers matched "
                f"of {len(cik_of)} universe tickers")
    return fn
