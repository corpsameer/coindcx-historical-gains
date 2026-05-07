import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

MARKETS_URL = "https://api.coindcx.com/exchange/v1/markets_details"
CANDLES_URL = "https://public.coindcx.com/market_data/candles"
CACHE_TTL_SECONDS = 6 * 60 * 60
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
MAX_WORKERS = 12

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_JSON = OUTPUT_DIR / "coindcx_market_snapshot.json"
OUTPUT_CSV = OUTPUT_DIR / "coindcx_market_snapshot.csv"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def request_with_retry(url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def load_markets() -> List[Dict]:
    response = request_with_retry(MARKETS_URL)
    if response is None:
        raise RuntimeError("Failed to fetch CoinDCX market details")
    data = response.json()
    return [m for m in data if m.get("status", "").lower() == "active"]


def classify_market(market: Dict) -> Optional[str]:
    symbol = str(market.get("symbol", ""))
    pair = str(market.get("pair", ""))
    base = str(market.get("base_currency_short_name", "")).upper()
    target = str(market.get("target_currency_short_name", "")).upper()

    # USDT futures commonly have symbols/pairs containing FUT or perpetual naming.
    futures_hints = ["FUT", "PERP", "USDTFUT", "B-", "S-"]
    if any(h in symbol.upper() for h in futures_hints) or any(h in pair.upper() for h in futures_hints):
        if "USDT" in symbol.upper() or base == "USDT" or target == "USDT":
            return "futures"

    # Conservative detection for USDT futures naming styles.
    if str(market.get("market", "")).lower() in {"futures", "future"} and ("USDT" in symbol.upper() or target == "USDT"):
        return "futures"

    if str(market.get("market", "")).lower() in {"spot", ""}:
        return "spot"

    # Fallback: treat unrecognized as spot so we do not silently lose active markets.
    return "spot"


def cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("-", "_")
    return CACHE_DIR / f"{safe}_{interval}.json"


def load_cached_candles(symbol: str, interval: str) -> Optional[List]:
    path = cache_path(symbol, interval)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_cached_candles(symbol: str, interval: str, payload: List) -> None:
    cache_path(symbol, interval).write_text(json.dumps(payload))


def fetch_candles(symbol: str, interval: str, limit: int) -> Optional[List[Dict]]:
    cached = load_cached_candles(symbol, interval)
    if cached is not None:
        return cached

    params = {"pair": symbol, "interval": interval, "limit": limit}
    response = request_with_retry(CANDLES_URL, params=params)
    if response is None:
        return None
    payload = response.json()
    if not isinstance(payload, list):
        return None
    save_cached_candles(symbol, interval, payload)
    return payload


def to_df(candles: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def pct_change(current: float, previous: float) -> Optional[float]:
    if previous in (None, 0) or pd.isna(previous) or pd.isna(current):
        return None
    return (current - previous) / previous * 100.0


def safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi.iloc[-1])


def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return safe_float(atr.iloc[-1])


def classify_trend(close: pd.Series) -> str:
    if len(close) < 30:
        return "sideways"
    ma7 = close.rolling(7).mean().iloc[-1]
    ma30 = close.rolling(30).mean().iloc[-1]
    latest = close.iloc[-1]
    slope = pct_change(ma7, close.rolling(7).mean().iloc[-8]) if len(close) >= 37 else 0
    if latest > ma7 > ma30 and (slope or 0) > 8:
        return "strong_uptrend"
    if latest > ma7 > ma30:
        return "uptrend"
    if latest < ma7 < ma30 and (slope or 0) < -8:
        return "strong_downtrend"
    if latest < ma7 < ma30:
        return "downtrend"
    return "sideways"


def classify_momentum(change_7d, change_30d, rvol, breakout_1d) -> str:
    if (change_7d or 0) < -8:
        return "breakdown"
    if (change_30d or 0) < 5 and (rvol or 0) < 1:
        return "base"
    if breakout_1d and 5 <= (change_7d or 0) < 20:
        return "early_expansion"
    if breakout_1d and (change_7d or 0) >= 20:
        return "active_expansion"
    if (change_30d or 0) >= 80:
        return "late_parabolic"
    return "base"


def build_symbol_snapshot(market: Dict, snapshot_time: str) -> Tuple[Optional[Dict], Optional[str]]:
    symbol = str(market.get("coindcx_name") or market.get("symbol") or "")
    market_type = classify_market(market)
    if not symbol or market_type not in {"spot", "futures"}:
        return None, symbol or "unknown"

    candles_1d = fetch_candles(symbol, "1d", 90)
    candles_4h = fetch_candles(symbol, "4h", 180)
    if not candles_1d or not candles_4h:
        return None, symbol

    df1 = to_df(candles_1d)
    df4 = to_df(candles_4h)
    if len(df1) < 31 or len(df4) < 43:
        return None, symbol

    c = df1["close"]
    v = df1["volume"].fillna(0)
    current_price = safe_float(c.iloc[-1])
    high_30d = safe_float(df1["high"].tail(30).max())
    low_30d = safe_float(df1["low"].tail(30).min())
    avg_volume_7d = safe_float(v.tail(7).mean())
    avg_volume_30d = safe_float(v.tail(30).mean())
    latest_volume = safe_float(v.iloc[-1])
    relative_volume = safe_float((avg_volume_7d / avg_volume_30d) if avg_volume_30d else None)

    change_7d = pct_change(c.iloc[-1], c.iloc[-8])
    change_30d = pct_change(c.iloc[-1], c.iloc[-31])
    breakout_1d = bool(current_price and high_30d and current_price >= high_30d * 0.995)
    near_30d_high = bool(current_price and high_30d and current_price >= high_30d * 0.98)
    new_30d_high = bool(len(df1) >= 31 and df1["high"].iloc[-1] >= df1["high"].tail(30).max())

    returns_1d = c.pct_change().dropna()
    vol_7d = safe_float(returns_1d.tail(7).std() * 100)
    vol_30d = safe_float(returns_1d.tail(30).std() * 100)
    compression = bool((vol_30d or 0) < 4 and (vol_7d or 0) < (vol_30d or 0))
    expansion = bool((vol_7d or 0) > (vol_30d or 0) * 1.2)

    c4 = df4["close"]
    high_7d_4h = safe_float(df4["high"].tail(42).max())
    breakout_4h = bool(c4.iloc[-1] >= df4["high"].tail(12).max() * 0.998)

    rsi = compute_rsi(c, 14)
    atr = compute_atr(df1, 14)
    atr_pct = safe_float((atr / current_price * 100) if atr and current_price else None)

    latest = df1.iloc[-1]
    body = abs(latest["close"] - latest["open"])
    range_ = max(latest["high"] - latest["low"], 1e-12)
    candle_body_strength = safe_float(body / range_ * 100)
    upper_wick = safe_float((latest["high"] - max(latest["open"], latest["close"])) / range_ * 100)
    lower_wick = safe_float((min(latest["open"], latest["close"]) - latest["low"]) / range_ * 100)

    momentum_stage = classify_momentum(change_7d, change_30d, relative_volume, breakout_1d)
    trend_state = classify_trend(c)

    score = 0.0
    score += min(max((change_7d or 0), 0), 30) * 0.7
    score += min(max(((relative_volume or 0) - 1) * 20, 0), 20)
    score += 10 if breakout_1d and near_30d_high else 0
    score += min(max(((vol_7d or 0) - (vol_30d or 0)) * 4, 0), 10)
    score += 10 if momentum_stage == "early_expansion" else 0
    score += 10 if compression and expansion else 0
    score += 10 if breakout_4h and (pct_change(c4.iloc[-1], c4.iloc[-7]) or 0) > 2 else 0
    score = max(0, min(100, round(score, 2)))

    if score >= 85:
        potential = "10x_plus"
    elif score >= 70:
        potential = "5x_to_10x"
    elif score >= 55:
        potential = "3x_to_5x"
    elif score >= 40:
        potential = "2x_to_3x"
    else:
        potential = "1.5x_to_2x"

    snap = {
        "market_type": market_type,
        "symbol": str(market.get("symbol") or ""),
        "pair": str(market.get("pair") or symbol),
        "current_price": current_price,
        "snapshot_time_utc": snapshot_time,
        "change_1d_percent": safe_float(pct_change(c.iloc[-1], c.iloc[-2])),
        "change_3d_percent": safe_float(pct_change(c.iloc[-1], c.iloc[-4])),
        "change_7d_percent": safe_float(change_7d),
        "change_14d_percent": safe_float(pct_change(c.iloc[-1], c.iloc[-15])),
        "change_30d_percent": safe_float(change_30d),
        "high_30d": high_30d,
        "low_30d": low_30d,
        "distance_from_30d_high_percent": safe_float(pct_change(current_price, high_30d)),
        "distance_from_30d_low_percent": safe_float(pct_change(current_price, low_30d)),
        "highest_volume_30d": safe_float(v.tail(30).max()),
        "latest_volume": latest_volume,
        "volume_change_7d_percent": safe_float(pct_change(avg_volume_7d, avg_volume_30d)),
        "avg_volume_7d": avg_volume_7d,
        "avg_volume_30d": avg_volume_30d,
        "relative_volume_7d_vs_30d": relative_volume,
        "volatility_7d_percent": vol_7d,
        "volatility_30d_percent": vol_30d,
        "change_4h_percent": safe_float(pct_change(c4.iloc[-1], c4.iloc[-2])),
        "change_12h_percent": safe_float(pct_change(c4.iloc[-1], c4.iloc[-4])),
        "change_24h_percent": safe_float(pct_change(c4.iloc[-1], c4.iloc[-7])),
        "high_7d_4h": high_7d_4h,
        "low_7d_4h": safe_float(df4["low"].tail(42).min()),
        "distance_from_7d_high_4h_percent": safe_float(pct_change(c4.iloc[-1], high_7d_4h)),
        "breakout_4h_detected": breakout_4h,
        "daily_trend_state": trend_state,
        "momentum_stage": momentum_stage,
        "breakout_1d_detected": breakout_1d,
        "near_30d_high": near_30d_high,
        "new_30d_high": new_30d_high,
        "compression_detected": compression,
        "expansion_detected": expansion,
        "rsi_14_daily": rsi,
        "atr_14_daily": atr,
        "atr_percent_daily": atr_pct,
        "candle_body_strength_latest": candle_body_strength,
        "upper_wick_percent_latest": upper_wick,
        "lower_wick_percent_latest": lower_wick,
        "resembles_explosive_setup_score": score,
        "max_x_gain_potential_class": potential,
    }
    return snap, None


def main() -> None:
    ensure_dirs()
    start = time.time()
    snapshot_time = datetime.now(timezone.utc).isoformat()

    markets = load_markets()
    enriched = []
    for m in markets:
        mtype = classify_market(m)
        if mtype in {"spot", "futures"}:
            enriched.append(m)

    total_spot = sum(1 for m in enriched if classify_market(m) == "spot")
    total_futures = sum(1 for m in enriched if classify_market(m) == "futures")

    snapshots: List[Dict] = []
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(build_symbol_snapshot, m, snapshot_time): m for m in enriched}
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Building snapshots"):
            snap, err_symbol = fut.result()
            if snap:
                snapshots.append(snap)
            elif err_symbol:
                failed.append(err_symbol)

    snapshots.sort(key=lambda x: x.get("resembles_explosive_setup_score", 0), reverse=True)

    result = {
        "snapshot_generated_at_utc": snapshot_time,
        "source": "CoinDCX only",
        "total_spot_symbols": total_spot,
        "total_futures_symbols": total_futures,
        "total_symbols_with_snapshot": len(snapshots),
        "failed_symbols": failed,
        "symbols": snapshots,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2))
    pd.DataFrame(snapshots).to_csv(OUTPUT_CSV, index=False)

    runtime = round(time.time() - start, 2)
    top20 = snapshots[:20]

    print("\n=== CoinDCX Snapshot Summary ===")
    print(f"Total markets found: {len(markets)}")
    print(f"Total spot markets: {total_spot}")
    print(f"Total futures markets: {total_futures}")
    print(f"Total successful snapshots: {len(snapshots)}")
    print(f"Total failed/skipped: {len(failed)}")
    print("Top 20 symbols by resembles_explosive_setup_score:")
    for row in top20:
        print(f"  {row['symbol']}: {row['resembles_explosive_setup_score']}")
    print(f"Runtime (seconds): {runtime}")


if __name__ == "__main__":
    main()
