"""DuckDB + Parquet data store — the robust replacement for loading giant CSVs into RAM.

The Sharadar bulk `stocks` CSV is ~1.2 GB. Parsing it into pandas every run is slow and
memory-hungry. Instead we convert it to Parquet **once** (columnar, compressed ~5-6x
smaller), then let DuckDB read only the columns/rows we need with SQL — streaming, low
memory, seconds not minutes.

Public helpers:
  ensure_parquet()          convert the raw CSVs to Parquet (idempotent)
  load_tickers()            the full tickers metadata (incl. delisted) as a DataFrame
  load_prices(universe)     {ticker: OHLCV df} for a set of tickers, read from Parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

from sterling.research import config

logger = logging.getLogger(__name__)


def csv_to_parquet(csv_path: Path, parquet_path: Path, force: bool = False) -> Path:
    """Convert a bulk CSV to compressed Parquet via DuckDB (streamed, low memory)."""
    if parquet_path.exists() and not force:
        return parquet_path
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — download the Sharadar bulk file first")
    logger.info(f"converting {csv_path.name} -> {parquet_path.name} (Parquet/ZSTD)...")
    tmp = config.DATA / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # Bound memory and spill to disk so multi-GB CSVs convert without OOM.
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute(
        f"COPY (SELECT * FROM read_csv_auto('{csv_path.as_posix()}', "
        f"SAMPLE_SIZE=200000, IGNORE_ERRORS=true)) "
        f"TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    con.close()
    logger.info(f"  wrote {parquet_path.name} "
                f"({parquet_path.stat().st_size / 1e6:.0f} MB, from {csv_path.stat().st_size / 1e6:.0f} MB CSV)")
    return parquet_path


def ensure_parquet(force: bool = False) -> None:
    """Make sure both bulk tables exist as Parquet."""
    csv_to_parquet(config.STOCKS_CSV, config.STOCKS_PARQUET, force)
    csv_to_parquet(config.TICKERS_CSV, config.TICKERS_PARQUET, force)


def load_tickers(parquet: Optional[Path] = None) -> pd.DataFrame:
    """Full tickers metadata (common stock, incl. delisted) with parsed dates."""
    parquet = parquet or config.TICKERS_PARQUET
    if not parquet.exists():
        csv_to_parquet(config.TICKERS_CSV, parquet)
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{parquet.as_posix()}') "
        f"WHERE category LIKE '%Common Stock%'"
    ).df()
    con.close()
    for c in ("firstpricedate", "lastpricedate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_prices(universe: Iterable[str], parquet: Optional[Path] = None,
                min_bars: int = config.MIN_BARS) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV df} for `universe`, read from Parquet via DuckDB.

    Only the needed tickers/columns are scanned (uses adjusted close). Far faster and
    lighter than a chunked CSV read, and the point-in-time series for delisted names end
    naturally at their last trading day.
    """
    parquet = parquet or config.STOCKS_PARQUET
    if not parquet.exists():
        csv_to_parquet(config.STOCKS_CSV, parquet)
    uni = list(dict.fromkeys(universe))

    tmp = config.DATA / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    out: dict[str, pd.DataFrame] = {}
    batch = 800  # bound peak memory over full (multi-decade) history
    for i in range(0, len(uni), batch):
        uni_df = pd.DataFrame({"ticker": uni[i:i + batch]})
        con.register("uni", uni_df)
        rows = con.execute(
            f"SELECT s.ticker, s.date, s.open, s.high, s.low, s.closeadj, s.volume "
            f"FROM read_parquet('{parquet.as_posix()}') s JOIN uni USING (ticker) "
            f"WHERE s.closeadj IS NOT NULL ORDER BY s.ticker, s.date"
        ).df()
        con.unregister("uni")
        for tk, g in rows.groupby("ticker", sort=False):
            idx = pd.DatetimeIndex(pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize())
            df = pd.DataFrame({
                "Open": g["open"].to_numpy(), "High": g["high"].to_numpy(),
                "Low": g["low"].to_numpy(), "Close": g["closeadj"].to_numpy(),
                "Volume": g["volume"].to_numpy(),
            }, index=idx)
            if len(df) >= min_bars:
                out[tk] = df
    con.close()
    return out


def download_bulk(table: str, years: str = "full", dest=None):
    """Download a Sharadar bulk table (stocks/tickers/...) to a CSV. Reusable by the
    cloud bootstrap. Reads the key from env; follows the redirect to the file."""
    import io, os, zipfile, requests
    from dotenv import dotenv_values
    key = (dotenv_values(".env").get("NASDAQ_API_KEY") or dotenv_values(".env").get("SHARADAR_API_KEY")
           or os.getenv("NASDAQ_API_KEY") or os.getenv("SHARADAR_API_KEY"))
    dest = Path(dest) if dest else config.BULK / f"{table}_{years}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://api.sharadar.com/v1.0/data/{table}",
                     params={"api_key": key, "years": years}, allow_redirects=True,
                     stream=True, timeout=3600)
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
    z = zipfile.ZipFile(buf)
    name = z.namelist()[0]
    z.extract(name, dest.parent)
    os.replace(dest.parent / name, dest)
    return dest
