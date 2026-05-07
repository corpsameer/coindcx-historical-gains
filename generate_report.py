#!/usr/bin/env python3
"""CoinDCX rolling 30-day chained-window winners report (Spot + USDT Futures)."""

from __future__ import annotations

import json
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
    session = requests.Session()
    retry = Retry(total=MAX_RETRIES, connect=MAX_RETRIES, read=MAX_RETRIES, status=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_with_explicit_retries(session: requests.Session, url: str, params: Optional[dict] = None) -> requests.Response:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < MAX_RETRIES:
            time.sleep((2**attempt) * BACKOFF_FACTOR)
    raise RuntimeError(f"Request failed after retries: {url} {params}. {last_exc}")


def fetch_markets(session: requests.Session) -> List[dict]:
    data = get_with_explicit_retries(session, MARKETS_URL).json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected markets format")
    return data


def is_active_market(m: dict) -> bool:
    return bool(m.get("active", False)) or str(m.get("status", "")).lower() in {"active", "enabled"}


def looks_like_futures(m: dict, symbol: str, pair: str) -> bool:
    blob = " ".join(str(m.get(k, "")).lower() for k in ["market_type", "market", "instrument_type", "segment", "contract_type", "name", "coindcx_name"])
    return any(x in blob for x in ["future", "futures", "perp", "perpetual", "swap"])


def split_markets(markets: List[dict]) -> Tuple[List[dict], List[dict]]:
    spot, futures = [], []
    for m in markets:
        if not isinstance(m, dict) or not is_active_market(m):
            continue
        pair = m.get("pair") or m.get("coindcx_name") or m.get("symbol")
        symbol = m.get("symbol") or m.get("coindcx_name") or pair
        if not pair or not symbol:
            continue
        quote = str(m.get("target_currency_short_name") or m.get("quote_currency_short_name") or m.get("quote_currency") or "").upper()
        fut = looks_like_futures(m, symbol, pair)
        if fut and (quote == "USDT" or "USDT" in str(symbol).upper() or "USDT" in str(pair).upper()):
            futures.append({**m, "pair": pair, "symbol": symbol})
        elif not fut:
            spot.append({**m, "pair": pair, "symbol": symbol})

    return spot, futures


def cache_path(market_type: str, pair: str) -> Path:
    return CACHE_DIR / f"{market_type}_{pair.replace('/', '_')}.json"


def parse_candles(raw) -> pd.DataFrame:
    rows = []
    if isinstance(raw, dict):
        for key in ["data", "candles", "result"]:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return pd.DataFrame(columns=["date", "open", "high"])
    for c in raw:
        try:
            if isinstance(c, dict):
                ts, o, h = c.get("time"), c.get("open"), c.get("high")
            elif isinstance(c, (list, tuple)) and len(c) >= 3:
                ts, o, h = c[0], c[1], c[2]
            else:
                continue
            d = pd.to_datetime(int(ts), unit="ms", utc=True).date()
            rows.append({"date": d, "open": float(o), "high": float(h)})
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high"])
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def fetch_symbol_df(session, market, market_type, start_ms, end_ms):
    symbol, pair = market.get("symbol", ""), market.get("pair", "")
    cp = cache_path(market_type, pair)
    try:
        if cp.exists():
            raw = json.loads(cp.read_text(encoding="utf-8"))
        else:
            raw = get_with_explicit_retries(session, CANDLES_URL, params={"pair": pair, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": 1000}).json()
            cp.write_text(json.dumps(raw), encoding="utf-8")
        if (isinstance(raw, list) and not raw) or raw is None:
            return symbol, pair, None, "empty candle data"
        df = parse_candles(raw)
        if df.empty:
            return symbol, pair, None, "invalid candle format"
        return symbol, pair, df, None
    except Exception as exc:
        return symbol, pair, None, str(exc)


def symbol_best_by_window(symbol: str, pair: str, df: pd.DataFrame, ws: pd.Timestamp) -> Optional[dict]:
    ws_d = ws.date()
    indexed = df.set_index("date")
    if ws_d not in indexed.index:
        return None
    start_price = float(indexed.loc[ws_d, "open"] if not isinstance(indexed.loc[ws_d, "open"], pd.Series) else indexed.loc[ws_d, "open"].iloc[0])
    if start_price <= 0:
        return None
    we = (ws + pd.Timedelta(days=WINDOW_DAYS)).date()
    wdf = df[(df["date"] >= ws_d) & (df["date"] <= we)]
    if wdf.empty:
        return None
    mr = wdf.loc[wdf["high"].idxmax()]
    max_price = float(mr["high"])
    return {"window_start_date": ws_d, "window_end_date": we, "symbol": symbol, "pair": pair, "start_price": start_price, "max_price": max_price, "max_date": mr["date"], "gain_percent": ((max_price - start_price) / start_price) * 100.0, "x_gain": max_price / start_price}


def process_group(session, markets, market_type, start_ms, end_ms, start_date: str, end_date: str):
    symbol_data, failed, processed = [], [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch_symbol_df, session, m, market_type, start_ms, end_ms) for m in markets]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{market_type} candles"):
            processed += 1
            symbol, pair, df, err = f.result()
            if err or df is None:
                failed.append(f"{market_type}:{symbol} ({err})")
                continue
            symbol_data.append((symbol, pair, df))

    # True chained windows: next window starts at previous winner max_date + 1 day.
    rows = []
    current = pd.Timestamp(start_date)
    hard_end = pd.Timestamp(end_date)
    while current <= hard_end:
        best = None
        for symbol, pair, df in symbol_data:
            row = symbol_best_by_window(symbol, pair, df, current)
            if row is None:
                continue
            if best is None or row["gain_percent"] > best["gain_percent"]:
                best = row

        if best is None:
            # no eligible symbol for this day; move to next day
            current += pd.Timedelta(days=1)
            continue

        best["market_type"] = market_type
        rows.append(best)
        current = pd.Timestamp(best["max_date"]) + pd.Timedelta(days=1)

    return rows, failed, processed


def export_winners(df, csv_path, xlsx_path):
    df = df.sort_values("window_start_date", ascending=True).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    pretty = df.rename(columns={"window_start_date": "Window Start Date", "window_end_date": "Window End Date", "symbol": "Symbol", "pair": "Pair", "start_price": "Start Price", "max_price": "Max Price", "max_date": "Max Date", "gain_percent": "Gain %", "x_gain": "X Gain"})[["Window Start Date", "Window End Date", "Symbol", "Pair", "Start Price", "Max Price", "Max Date", "Gain %", "X Gain"]]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pretty.to_excel(writer, sheet_name="Winners", index=False)


def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    all_markets = fetch_markets(session)
    spot, futures = split_markets(all_markets)
    print("Classification preview: total", len(all_markets), "spot", len(spot), "futures", len(futures))

    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts = pd.Timestamp(END_DATE, tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int((end_ts + pd.Timedelta(days=1)).timestamp() * 1000) - 1

    spot_rows, spot_failed, spot_processed = process_group(session, spot, "spot", start_ms, end_ms, START_DATE, END_DATE)
    fut_rows, fut_failed, fut_processed = process_group(session, futures, "futures", start_ms, end_ms, START_DATE, END_DATE)

    cols = ["market_type", "window_start_date", "window_end_date", "symbol", "pair", "start_price", "max_price", "max_date", "gain_percent", "x_gain"]
    spot_df = pd.DataFrame(spot_rows, columns=cols)
    fut_df = pd.DataFrame(fut_rows, columns=cols)

    export_winners(spot_df, OUTPUT_DIR / "spot_window_winners.csv", OUTPUT_DIR / "spot_window_winners.xlsx")
    export_winners(fut_df, OUTPUT_DIR / "futures_window_winners.csv", OUTPUT_DIR / "futures_window_winners.xlsx")

    with pd.ExcelWriter(OUTPUT_DIR / "all_window_winners.xlsx", engine="openpyxl") as writer:
        spot_df.sort_values("window_start_date").to_excel(writer, sheet_name="Spot", index=False)
        fut_df.sort_values("window_start_date").to_excel(writer, sheet_name="Futures", index=False)

    failed = spot_failed + fut_failed
    print("\n=== CoinDCX Rolling 30-Day Window Winner Summary ===")
    print(f"Spot markets found: {len(spot)}")
    print(f"USDT futures markets found: {len(futures)}")
    print(f"Total symbols processed: {spot_processed + fut_processed}")
    print(f"Total skipped symbols: {len(failed)}")
    print(f"Total rows generated: {len(spot_df) + len(fut_df)}")
    print(f"Runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
