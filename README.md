# CoinDCX Rolling 30-Day Gainers Report

Lightweight Python-only reporting script to generate CoinDCX-specific rolling 30-day highest gainers for Spot and USDT Futures using CoinDCX APIs only.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_report.py
```

## Output files

The script auto-creates `cache/` and `output/` and generates:

- `output/spot_full.csv` — all rolling rows for spot markets
- `output/futures_full.csv` — all rolling rows for USDT futures markets
- `output/spot_top_gainers.xlsx` — top 500 spot rows by `gain_percent`
- `output/futures_top_gainers.xlsx` — top 500 futures rows by `gain_percent`

Raw candle payloads are cached per market in `cache/` as JSON and reused on reruns.

## Notes / known limitation

Exact historical listing/delisting status for each past date may not be fully available from active market details. The script therefore uses currently available active CoinDCX markets and gracefully skips symbols without sufficient historical data.
