# CoinDCX Rolling 30-Day Window Winners Report

Python-only script that uses CoinDCX APIs to compute rolling 30-day winners:
- exactly **one Spot winner per window**
- exactly **one USDT Futures winner per window**

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_report.py
```

## Outputs

Auto-creates `cache/` and `output/` and writes:

- `output/spot_window_winners.csv`
- `output/futures_window_winners.csv`
- `output/spot_window_winners.xlsx`
- `output/futures_window_winners.xlsx`
- `output/all_window_winners.xlsx` (sheets: `Spot`, `Futures`)

Rows are sorted by `window_start_date` ascending.

## Known limitation

Exact historical listing/delisting status for each past date may not be fully available from active market details. The script uses currently active CoinDCX markets and skips symbols without sufficient historical data.

## Candle endpoint note

`https://public.coindcx.com/market_data/candles` will return `400 Invalid Request` if query parameters are missing.
The script always sends required params (`pair`, `interval`) and optional range params (`startTime`, `endTime`) as documented.
