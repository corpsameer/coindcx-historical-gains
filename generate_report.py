#!/usr/bin/env python3
"""
CoinDCX rolling 30-day highest gainers report generator.

This script fetches currently active CoinDCX markets, splits into Spot and USDT Futures,
downloads 1D candles for each market pair, computes rolling 30-day maximum gain windows,
and exports CSV + Excel reports.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

MARKETS_URL = "https://api.coindcx.com/exchange/v1/markets_details"
CANDLES_URL = "https://public.coindcx.com/market_data/candles"

START_DATE = "2025-04-01"
END_DATE = "2026-04-30"
WINDOW_DAYS = 30
MAX_WORKERS = 16
TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"


def make_session() -> requests.Session:
    """Build an HTTP session with retry + backoff for resilient API calls."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_with_explicit_retries(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    timeout: int = TIMEOUT_SECONDS,
) -> requests.Response:
    """Explicit retry wrapper with exponential backoff and timeout handling."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < MAX_RETRIES:
            sleep_for = (2**attempt) * BACKOFF_FACTOR
            time.sleep(sleep_for)

    raise RuntimeError(f"Request failed after retries for {url} params={params}. {last_exc}")


def fetch_markets(session: requests.Session) -> List[dict]:
    resp = get_with_explicit_retries(session, MARKETS_URL)
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected markets response format; expected list")
    return data


def split_markets(markets: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Filter active markets and split to Spot vs USDT Futures.

    CoinDCX markets_details can include many market flavors. We classify:
    - Spot: active markets that are not futures
    - USDT Futures: active futures where quote currency is USDT

    We preserve both the user-facing symbol and internal pair used by candle API.
    """
    spot_markets = []
    futures_markets = []

    for m in markets:
        if not isinstance(m, dict):
            continue
        if not m.get("active", False):
            continue

        pair = m.get("pair")
        symbol = m.get("symbol")
        if not pair or not symbol:
            continue

        market_type_raw = str(m.get("market_type", "")).lower()
        quote_currency = str(m.get("target_currency_short_name", "")).upper()

        is_futures = "futures" in market_type_raw
        if is_futures and quote_currency == "USDT":
            futures_markets.append(m)
        elif not is_futures:
            spot_markets.append(m)

    return spot_markets, futures_markets


def cache_path(market_type: str, pair: str) -> Path:
    safe_pair = pair.replace("/", "_")
    return CACHE_DIR / f"{market_type}_{safe_pair}.json"


def parse_candles_to_df(raw_data: list) -> pd.DataFrame:
    """
    Parse CoinDCX candles response to normalized DataFrame.

    Candle API can return a list of arrays. Expected order: [time, open, high, low, close, volume].
    We parse only fields needed for rolling gain logic.
    """
    rows = []
    for item in raw_data:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            ts_ms = int(item[0])
            o = float(item[1])
            h = float(item[2])
        except (TypeError, ValueError):
            continue

        dt = pd.to_datetime(ts_ms, unit="ms", utc=True).date()
        rows.append({"date": dt, "open": o, "high": h})

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high"])

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    return df


def fetch_candles_for_market(
    market: dict,
    market_type: str,
    session: requests.Session,
    start_ms: int,
    end_ms: int,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    symbol = market.get("symbol", "")
    pair = market.get("pair", "")
    cpath = cache_path(market_type, pair)

    try:
        if cpath.exists():
            with cpath.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            params = {
                "pair": pair,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = get_with_explicit_retries(session, CANDLES_URL, params=params)
            raw = resp.json()
            with cpath.open("w", encoding="utf-8") as f:
                json.dump(raw, f)

        if not isinstance(raw, list) or len(raw) == 0:
            return symbol, None, "empty candle data"

        df = parse_candles_to_df(raw)
        if df.empty:
            return symbol, None, "invalid candle format"

        return symbol, df, None
    except Exception as exc:
        return symbol, None, str(exc)


def compute_rolling_rows(df: pd.DataFrame, market_type: str, symbol: str, pair: str) -> List[dict]:
    """
    Rolling 30-day logic:
    for each start date i, evaluate [i, i+29] calendar days by date filter,
    use start open and max high within that date interval.
    """
    if df.empty:
        return []

    rows = []
    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D").date

    indexed = df.set_index("date")

    for start_date in all_dates:
        if start_date not in indexed.index:
            continue
        start_price = indexed.loc[start_date, "open"]
        if isinstance(start_price, pd.Series):
            start_price = float(start_price.iloc[0])
        else:
            start_price = float(start_price)

        if start_price <= 0:
            continue

        window_end = start_date + pd.Timedelta(days=WINDOW_DAYS - 1)
        wdf = df[(df["date"] >= start_date) & (df["date"] <= window_end.date())]
        if wdf.empty:
            continue

        max_idx = wdf["high"].idxmax()
        max_row = wdf.loc[max_idx]
        max_price = float(max_row["high"])
        max_date = max_row["date"]

        gain_percent = ((max_price - start_price) / start_price) * 100.0
        x_gain = max_price / start_price

        rows.append(
            {
                "market_type": market_type,
                "symbol": symbol,
                "pair": pair,
                "start_date": start_date,
                "max_date": max_date,
                "start_price": start_price,
                "max_price": max_price,
                "gain_percent": gain_percent,
                "x_gain": x_gain,
            }
        )

    return rows


def export_reports(df: pd.DataFrame, full_csv: Path, top_xlsx: Path) -> None:
    df_sorted = df.sort_values("gain_percent", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(full_csv, index=False)

    top500 = df_sorted.head(500).copy()
    pretty = top500.rename(
        columns={
            "symbol": "Symbol",
            "pair": "Pair",
            "start_date": "Start Date",
            "max_date": "Max Date",
            "start_price": "Start Price",
            "max_price": "Max Price",
            "gain_percent": "Gain %",
            "x_gain": "X Gain",
        }
    )[
        [
            "Symbol",
            "Pair",
            "Start Date",
            "Max Date",
            "Start Price",
            "Max Price",
            "Gain %",
            "X Gain",
        ]
    ]

    with pd.ExcelWriter(top_xlsx, engine="openpyxl") as writer:
        pretty.to_excel(writer, sheet_name="Top Gainers", index=False)
        ws = writer.sheets["Top Gainers"]

        for cell in ws["G"][1:]:
            cell.number_format = "0.00"
        for cell in ws["H"][1:]:
            cell.number_format = "0.00"
        for col in ("E", "F"):
            for cell in ws[col][1:]:
                cell.number_format = "0.00000000"


def process_market_group(
    session: requests.Session,
    markets: List[dict],
    market_type: str,
    start_ms: int,
    end_ms: int,
) -> Tuple[List[dict], List[str], int]:
    rows: List[dict] = []
    failed_symbols: List[str] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(fetch_candles_for_market, m, market_type, session, start_ms, end_ms): m
            for m in markets
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{market_type} candles"):
            market = futures[future]
            symbol = market.get("symbol", "")
            pair = market.get("pair", "")

            processed += 1
            sym, df, err = future.result()
            if err or df is None:
                failed_symbols.append(f"{market_type}:{sym} ({err})")
                continue

            sym_rows = compute_rolling_rows(df, market_type, symbol, pair)
            if not sym_rows:
                failed_symbols.append(f"{market_type}:{sym} (insufficient historical data)")
                continue
            rows.extend(sym_rows)

    return rows, failed_symbols, processed


def main() -> None:
    start_time = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    all_markets = fetch_markets(session)
    spot_markets, futures_markets = split_markets(all_markets)

    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000) - 1

    spot_rows, spot_failed, spot_processed = process_market_group(
        session, spot_markets, "spot", start_ms, end_ms
    )
    fut_rows, fut_failed, fut_processed = process_market_group(
        session, futures_markets, "futures", start_ms, end_ms
    )

    spot_df = pd.DataFrame(spot_rows)
    fut_df = pd.DataFrame(fut_rows)

    if spot_df.empty:
        spot_df = pd.DataFrame(
            columns=[
                "market_type",
                "symbol",
                "pair",
                "start_date",
                "max_date",
                "start_price",
                "max_price",
                "gain_percent",
                "x_gain",
            ]
        )
    if fut_df.empty:
        fut_df = spot_df.copy()

    export_reports(spot_df, OUTPUT_DIR / "spot_full.csv", OUTPUT_DIR / "spot_top_gainers.xlsx")
    export_reports(fut_df, OUTPUT_DIR / "futures_full.csv", OUTPUT_DIR / "futures_top_gainers.xlsx")

    failed = spot_failed + fut_failed
    total_rows = len(spot_df) + len(fut_df)
    runtime = time.time() - start_time

    print("\n=== CoinDCX Rolling 30-Day Report Summary ===")
    print(f"Spot markets found: {len(spot_markets)}")
    print(f"USDT futures markets found: {len(futures_markets)}")
    print(f"Total symbols processed: {spot_processed + fut_processed}")
    print(f"Total skipped symbols: {len(failed)}")
    print(f"Total rows generated: {total_rows}")
    print(f"Failed symbols list ({len(failed)}):")
    for s in failed:
        print(f"  - {s}")
    print(f"Runtime: {runtime:.2f}s")


if __name__ == "__main__":
    main()
