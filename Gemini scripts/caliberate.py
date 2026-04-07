import polars as pl
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
REPO_ROOT = Path(r"C:\prediction-market-analysis")
KALSHI_DATA = REPO_ROOT / "data" / "kalshi"
TICKER_SEARCH = "BTC" # Broader search to ensure we catch all regimes
NUMERIC_COLS = ["yes_price", "no_price", "yes_bid", "yes_ask", "no_bid", "no_ask", "last_price", "count", "strike_price", "price"]

def load_folder_with_force_cast(folder_path):
    if not folder_path.exists():
        file_path = folder_path.with_suffix(".parquet")
        if file_path.exists():
            return pl.read_parquet(file_path)
        raise FileNotFoundError(f"Path not found: {folder_path}")

    files = list(folder_path.glob("*.parquet"))
    print(f"Reading {len(files)} files from {folder_path.name}...")
    
    dfs = []
    for f in files:
        try:
            df_part = pl.read_parquet(f)
            rename_map = {"ts": "created_time", "price": "yes_price"}
            df_part = df_part.rename({k: v for k, v in rename_map.items() if k in df_part.columns and v not in df_part.columns})
            cols_to_cast = [c for c in NUMERIC_COLS if c in df_part.columns]
            df_part = df_part.with_columns([pl.col(c).cast(pl.Float64) for c in cols_to_cast])
            if "result" in df_part.columns:
                df_part = df_part.with_columns(pl.col("result").cast(pl.String))
            dfs.append(df_part)
        except Exception as e:
            print(f"⚠️ Skipping {f.name}: {e}")
    return pl.concat(dfs, how="diagonal")

def run_calibration():
    try:
        print("🚀 Loading data...")
        markets = load_folder_with_force_cast(KALSHI_DATA / "markets")
        trades = load_folder_with_force_cast(KALSHI_DATA / "trades")

        btc_markets = markets.filter(pl.col("ticker").str.contains(TICKER_SEARCH))
        df = trades.join(btc_markets.select(["ticker", "result", "close_time"]), on="ticker")
        
        # Feature Extraction
        df = df.with_columns([
            (pl.col("result").str.to_lowercase() == "yes").cast(pl.Int8).alias("actual_outcome"),
            ((pl.col("close_time").cast(pl.Datetime) - pl.col("created_time").cast(pl.Datetime)).dt.total_seconds() / 60.0).alias("tau_minutes"),
            (pl.col("yes_price") / 100.0).alias("market_prob")
        ])

        # FILTER FOR 0-15 MINUTE HORIZON ONLY
        df = df.filter((pl.col("tau_minutes") >= 0) & (pl.col("tau_minutes") <= 16))
        print(f"🧪 Analyzing {len(df):,} price points in the 15m horizon...")

        # Analysis by TTE
        df = df.with_columns([pl.col("tau_minutes").round(0).cast(pl.Int32).alias("tte_bucket")])
        analysis = df.group_by("tte_bucket").agg([
            pl.col("actual_outcome").mean().alias("realized_win_rate"),
            pl.col("market_prob").mean().alias("avg_market_price"),
            pl.len().alias("sample_size")
        ]).filter(pl.col("sample_size") > 100).sort("tte_bucket")

        # GENERATE MARKDOWN CONTENT
        md_content = "# KXBTC15M Calibration Report\n\n"
        md_content += f"- **Source Data Points:** {len(df):,}\n"
        md_content += "- **Analysis Horizon:** 0-15 minutes to expiry\n"
        md_content += "- **Target:** Bias correction for `fair_yes_prob` in `indicators.rs`\n\n"
        
        md_content += "## Calibration Table\n\n"
        md_content += "| TTE (min) | Market Prob | Realized Win% | Bias (Rust Offset) | Sample Size |\n"
        md_content += "|-----------|-------------|---------------|-------------------|-------------|\n"
        
        for row in analysis.to_dicts():
            bias = row['realized_win_rate'] - row['avg_market_price']
            md_content += f"| {row['tte_bucket']:>9} | {row['avg_market_price']:>11.4f} | {row['realized_win_rate']:>13.4f} | {bias:>+17.4f} | {row['sample_size']:>11,} |\n"

        md_content += "\n## Implementation Logic for Coding Engine\n\n"
        md_content += "```rust\n// Apply this in indicators.rs to the fair_yes_prob\nlet calibration_offset = match tau_minutes.round() as i32 {\n"
        for row in analysis.to_dicts():
            bias = row['realized_win_rate'] - row['avg_market_price']
            md_content += f"    {row['tte_bucket']} => {bias:.4f},\n"
        md_content += "    _ => 0.0,\n};\n\nlet calibrated_yes_prob = (fair_yes_prob + calibration_offset).clamp(0.01, 0.99);\n```\n"

        # WRITE TO FILE
        with open("result.md", "w") as f:
            f.write(md_content)
        
        print(f"✅ Success! Report generated in {Path.cwd() / 'result.md'}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_calibration()