from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import Booster
from pysr import PySRRegressor
from scipy.stats import norm, t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SMARTERPRED_PATH = Path(__file__).with_name("smarterpred.py")
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("model_tournament_output") / f"{DEFAULT_SERIES}_all_rows"
ML_MODEL_LABELS = {
    "Machine Learning (Logistic Regression)",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "PySR (Symbolic Regression)",
}
LIGHTGBM_MODEL_NAME = "7. LightGBM"
LIGHTGBM_MODEL_LABEL = "LightGBM"


@dataclass(frozen=True)
class StrategyConfig:
    starting_capital: float
    contracts_per_trade: int
    edge_threshold_cents: float
    standard_gaussian_entry_mode: str
    standard_gaussian_z_threshold: float
    fee_mode: str
    fee_per_contract_cents: float
    slippage_pct: float
    model_scope: str
    min_tau_minutes: float
    max_tau_minutes: float
    reserve_cash_pct: float
    enforce_capital_constraint: bool
    allow_stacking: bool

    @property
    def edge_threshold(self) -> float:
        return self.edge_threshold_cents / 100.0

    @property
    def fee_per_contract(self) -> float:
        return self.fee_per_contract_cents / 100.0

    @property
    def slippage(self) -> float:
        return self.slippage_pct / 100.0


@dataclass
class OpenPosition:
    ticker: str
    close_time: pd.Timestamp
    entry_cost: float
    payout: float
    fees: float
    side: str
    hold_minutes: float


def load_smarterpred_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smarterpred_module", SMARTERPRED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load smarterpred module from {SMARTERPRED_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_artifacts_dir(artifacts_dir: str | None, series_ticker: str) -> Path:
    if artifacts_dir:
        return Path(artifacts_dir)
    return Path(__file__).with_name("model_tournament_output") / f"{series_ticker}_all_rows"


def select_pysr_run_directory(artifacts_dir: Path) -> Path:
    metrics_path = artifacts_dir / "pysr" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_params = metrics["best_params"]

    for candidate in metrics["candidates"]:
        if candidate["params"] == best_params:
            return artifacts_dir / "pysr_runs" / f"candidate_{candidate['candidate']:02d}"

    raise ValueError(f"Could not identify the winning PySR candidate from {metrics_path}")


def load_trade_frame_with_metadata(
    smarterpred: ModuleType,
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
) -> pd.DataFrame:
    trade_files, skipped_files = smarterpred.collect_trade_files(trades_path)

    print(f"Using markets file: {markets_path}")
    print(f"Using trades path: {trades_path}")
    print(f"Loaded {len(trade_files):,} non-empty trade parquet files")
    if skipped_files:
        print(f"Skipped {skipped_files:,} empty or unreadable trade parquet files")

    con = duckdb.connect()
    smarterpred.register_markets_view(con, "markets_source", markets_path)
    con.from_parquet(trade_files, union_by_name=True).create_view("trades_source", replace=True)

    df = con.execute(
        f"""
        WITH target_markets AS (
            SELECT
                ticker,
                LOWER(CAST(result AS VARCHAR)) AS result,
                CAST(close_time AS TIMESTAMPTZ) AS close_time
            FROM markets_source
            WHERE ticker LIKE '{series_ticker}-%'
        ),
        trade_points AS (
            SELECT
                t.ticker,
                CAST(t.created_time AS TIMESTAMPTZ) AS created_time,
                m.close_time,
                LEAST(GREATEST(CAST(t.yes_price AS DOUBLE) / 100.0, 0.01), 0.99) AS market_prob,
                CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS actual_outcome,
                DATE_DIFF(
                    'second',
                    CAST(t.created_time AS TIMESTAMPTZ),
                    m.close_time
                ) / 60.0 AS tau_minutes
            FROM trades_source t
            INNER JOIN target_markets m USING (ticker)
            WHERE t.yes_price IS NOT NULL
        ),
        filtered AS (
            SELECT
                ticker,
                created_time,
                close_time,
                market_prob,
                actual_outcome,
                tau_minutes,
                COALESCE(
                    market_prob - LAG(market_prob) OVER (
                        PARTITION BY ticker
                        ORDER BY created_time
                    ),
                    0.0
                ) AS price_momentum
            FROM trade_points
            WHERE tau_minutes > 0 AND tau_minutes <= 15
        )
        SELECT
            ticker,
            created_time,
            close_time,
            actual_outcome,
            market_prob,
            tau_minutes,
            price_momentum
        FROM filtered
        """
    ).fetchdf()
    con.close()

    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df = df.sort_values(["created_time", "ticker"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} tradable rows for profitability evaluation")
    return df


def build_profitability_dataset(
    smarterpred: ModuleType,
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    sample_size: int,
) -> dict[str, object]:
    df = load_trade_frame_with_metadata(smarterpred, series_ticker, markets_path, trades_path)

    actuals = df["actual_outcome"].to_numpy(dtype=np.int8)
    eval_indices = smarterpred.stratified_sample_indices(actuals, sample_size, smarterpred.ML_RANDOM_SEED)
    df_eval = df.iloc[eval_indices].reset_index(drop=True).copy()
    df_eval["eval_row_id"] = np.arange(len(df_eval), dtype=np.int64)

    market_probs_eval = df_eval["market_prob"].to_numpy(dtype=np.float32)
    tau_eval = df_eval["tau_minutes"].to_numpy(dtype=np.float32)
    momentum_eval = df_eval["price_momentum"].to_numpy(dtype=np.float32)
    actuals_eval = df_eval["actual_outcome"].to_numpy(dtype=np.int8)
    feature_matrix = smarterpred.build_feature_matrix(market_probs_eval, tau_eval, momentum_eval)

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
        market_prob_train_val,
        market_prob_test,
        tau_train_val,
        tau_test,
        momentum_train_val,
        momentum_test,
        ticker_train_val,
        ticker_test,
        created_time_train_val,
        created_time_test,
        close_time_train_val,
        close_time_test,
        row_id_train_val,
        row_id_test,
    ) = train_test_split(
        feature_matrix,
        actuals_eval,
        market_probs_eval,
        tau_eval,
        momentum_eval,
        df_eval["ticker"].to_numpy(),
        df_eval["created_time"].to_numpy(),
        df_eval["close_time"].to_numpy(),
        df_eval["eval_row_id"].to_numpy(),
        test_size=0.2,
        random_state=smarterpred.ML_RANDOM_SEED,
        stratify=actuals_eval,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=smarterpred.ML_RANDOM_SEED,
        stratify=y_train_val,
    )

    test_frame = pd.DataFrame(
        {
            "ticker": ticker_test,
            "created_time": pd.to_datetime(created_time_test, utc=True),
            "close_time": pd.to_datetime(close_time_test, utc=True),
            "actual_outcome": y_test,
            "market_prob": market_prob_test,
            "tau_minutes": tau_test,
            "price_momentum": momentum_test,
            "test_row_id": row_id_test,
        }
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "market_prob_test": market_prob_test,
        "tau_test": tau_test,
        "momentum_test": momentum_test,
        "test_frame": test_frame,
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
        "sample_rows": len(df_eval),
    }


def load_saved_predictions(
    smarterpred: ModuleType,
    artifacts_dir: Path,
    splits: dict[str, object],
    model_scope: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    market_prob_test = splits["market_prob_test"]
    momentum_test = splits["momentum_test"]

    predictions: dict[str, np.ndarray] = {}

    print("Loading LightGBM artifact...")
    lightgbm_model = Booster(model_file=str(artifacts_dir / "lightgbm" / "model.txt"))
    predictions[LIGHTGBM_MODEL_NAME] = smarterpred.clean_probabilities(lightgbm_model.predict(X_test))

    if model_scope != "lightgbm":
        z_implied_test = norm.ppf(market_prob_test)
        predictions["1. Standard Gaussian (The Market)"] = smarterpred.clean_probabilities(market_prob_test)
        predictions["2. Fat Tails (Student-T, df=5)"] = smarterpred.clean_probabilities(t.cdf(z_implied_test, df=5))
        predictions["3. Extreme Fat Tails (Student-T, df=3)"] = smarterpred.clean_probabilities(t.cdf(z_implied_test, df=3))
        predictions["4. Momentum-Adjusted Gaussian"] = smarterpred.clean_probabilities(
            predictions["1. Standard Gaussian (The Market)"] + (np.sign(momentum_test) * 0.02)
        )

        print("Training Logistic Regression for profitability evaluation...")
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        predictions["5. Machine Learning (Logistic Regression)"] = smarterpred.clean_probabilities(
            lr_model.predict_proba(X_test)[:, 1]
        )

        print("Loading XGBoost artifact...")
        xgb_model = XGBClassifier()
        xgb_model.load_model(artifacts_dir / "xgboost" / "model.json")
        predictions["6. Gradient Boosted Trees (XGBoost)"] = smarterpred.clean_probabilities(
            xgb_model.predict_proba(X_test)[:, 1]
        )

        print("Loading CatBoost artifact...")
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(str(artifacts_dir / "catboost" / "model.cbm"))
        predictions["8. CatBoost"] = smarterpred.clean_probabilities(catboost_model.predict_proba(X_test)[:, 1])

        print("Loading PySR artifact...")
        pysr_model = PySRRegressor.from_file(run_directory=select_pysr_run_directory(artifacts_dir))
        predictions["9. PySR (Symbolic Regression)"] = smarterpred.clean_probabilities(pysr_model.predict(X_test))

    overall_rows = []
    for model_name, model_predictions in predictions.items():
        overall_rows.append(
            {
                "model": smarterpred.MODEL_NAME_TO_LABEL[model_name],
                "log_loss": smarterpred.binary_log_loss(y_test, model_predictions),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("log_loss").reset_index(drop=True)
    print("\nOverall held-out log-loss (reconstructed from artifacts):")
    print(overall_df.to_string(index=False))
    overall_log_loss = dict(zip(overall_df["model"], overall_df["log_loss"]))

    return predictions, overall_log_loss


def filter_predictions_by_scope(
    predictions: dict[str, np.ndarray],
    overall_log_loss: dict[str, float],
    smarterpred: ModuleType,
    model_scope: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if model_scope == "all":
        return predictions, overall_log_loss
    if model_scope == "lightgbm":
        return {LIGHTGBM_MODEL_NAME: predictions[LIGHTGBM_MODEL_NAME]}, {
            LIGHTGBM_MODEL_LABEL: overall_log_loss[LIGHTGBM_MODEL_LABEL]
        }
    if model_scope != "ml":
        raise ValueError(f"Unsupported model scope: {model_scope}")

    filtered_predictions: dict[str, np.ndarray] = {}
    filtered_log_loss: dict[str, float] = {}
    for model_name, model_predictions in predictions.items():
        model_label = smarterpred.MODEL_NAME_TO_LABEL[model_name]
        if model_label in ML_MODEL_LABELS:
            filtered_predictions[model_name] = model_predictions
            filtered_log_loss[model_label] = overall_log_loss[model_label]
    return filtered_predictions, filtered_log_loss


def update_drawdown(equity: float, peak: float, max_drawdown: float, max_drawdown_pct: float) -> tuple[float, float, float]:
    peak = max(peak, equity)
    drawdown = peak - equity
    drawdown_pct = 0.0 if peak <= 0 else drawdown / peak
    max_drawdown = max(max_drawdown, drawdown)
    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    return peak, max_drawdown, max_drawdown_pct


def choose_trade_side(model_prob: float, market_prob: float, threshold: float) -> str | None:
    edge = model_prob - market_prob
    if edge >= threshold:
        return "YES"
    if edge <= -threshold:
        return "NO"
    return None


def choose_standard_gaussian_trade_side(
    market_prob: float,
    config: StrategyConfig,
) -> str | None:
    if config.standard_gaussian_entry_mode == "edge":
        return choose_trade_side(market_prob, market_prob, config.edge_threshold)
    if config.standard_gaussian_entry_mode == "zscore":
        z_value = float(norm.ppf(market_prob))
        if z_value >= config.standard_gaussian_z_threshold:
            return "YES"
        if z_value <= -config.standard_gaussian_z_threshold:
            return "NO"
        return None
    raise ValueError(f"Unsupported standard gaussian entry mode: {config.standard_gaussian_entry_mode}")


def apply_adverse_slippage(entry_price: float, slippage: float) -> float:
    if slippage <= 0:
        return entry_price
    return min(0.999999, entry_price * (1.0 + slippage))


def calculate_fees(entry_price: float, config: StrategyConfig) -> float:
    if config.fee_mode == "flat":
        return config.fee_per_contract * config.contracts_per_trade
    if config.fee_mode == "kalshi":
        raw_fee = 0.07 * config.contracts_per_trade * entry_price * (1.0 - entry_price)
        return np.ceil(raw_fee * 100.0) / 100.0
    raise ValueError(f"Unsupported fee mode: {config.fee_mode}")


def simulate_profitability_for_model(
    model_label: str,
    test_frame: pd.DataFrame,
    model_predictions: np.ndarray,
    config: StrategyConfig,
) -> dict[str, float | int | str]:
    frame = test_frame.copy()
    frame["model_prob"] = model_predictions
    frame = frame.sort_values(["created_time", "test_row_id"]).reset_index(drop=True)

    available_cash = config.starting_capital
    open_cost_basis = 0.0
    peak_equity = config.starting_capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    open_positions: list[tuple[pd.Timestamp, int, OpenPosition]] = []
    active_tickers: set[str] = set()

    total_trades = 0
    yes_trades = 0
    no_trades = 0
    wins = 0
    skipped_capital = 0
    skipped_reserve = 0
    skipped_stacking = 0
    total_fees = 0.0
    gross_pnl = 0.0
    net_pnl = 0.0
    turnover = 0.0
    total_hold_minutes = 0.0
    sequence = 0

    def settle_positions_up_to(cutoff_time: pd.Timestamp) -> None:
        nonlocal available_cash, open_cost_basis, peak_equity, max_drawdown, max_drawdown_pct
        nonlocal wins, gross_pnl, net_pnl, total_hold_minutes

        while open_positions and open_positions[0][0] <= cutoff_time:
            _, _, position = heapq.heappop(open_positions)
            open_cost_basis -= position.entry_cost
            available_cash += position.payout
            trade_gross_pnl = position.payout - position.entry_cost
            trade_net_pnl = trade_gross_pnl - position.fees
            gross_pnl += trade_gross_pnl
            net_pnl += trade_net_pnl
            total_hold_minutes += position.hold_minutes
            if trade_net_pnl > 0:
                wins += 1
            active_tickers.discard(position.ticker)
            equity = available_cash + open_cost_basis
            peak_equity, max_drawdown, max_drawdown_pct = update_drawdown(
                equity,
                peak_equity,
                max_drawdown,
                max_drawdown_pct,
            )

    for row in frame.itertuples(index=False):
        settle_positions_up_to(row.created_time)

        if not config.allow_stacking and row.ticker in active_tickers:
            skipped_stacking += 1
            continue

        if model_label == "Standard Gaussian":
            side = choose_standard_gaussian_trade_side(row.market_prob, config)
        else:
            side = choose_trade_side(row.model_prob, row.market_prob, config.edge_threshold)
        if side is None:
            continue

        if side == "YES":
            base_entry_price = float(row.market_prob)
            payout = float(row.actual_outcome)
            yes_trades += 1
        else:
            base_entry_price = float(1.0 - row.market_prob)
            payout = float(1 - row.actual_outcome)
            no_trades += 1

        entry_price = apply_adverse_slippage(base_entry_price, config.slippage)
        entry_cost = entry_price * config.contracts_per_trade
        fees = calculate_fees(entry_price, config)
        cash_required = entry_cost + fees

        if config.enforce_capital_constraint and available_cash + 1e-12 < cash_required:
            skipped_capital += 1
            if side == "YES":
                yes_trades -= 1
            else:
                no_trades -= 1
            continue

        equity_before_trade = available_cash + open_cost_basis
        max_deployed_capital = (1.0 - (config.reserve_cash_pct / 100.0)) * equity_before_trade
        if (open_cost_basis + entry_cost) - max_deployed_capital > 1e-12:
            skipped_reserve += 1
            if side == "YES":
                yes_trades -= 1
            else:
                no_trades -= 1
            continue

        total_trades += 1
        total_fees += fees
        turnover += entry_cost
        available_cash -= cash_required
        open_cost_basis += entry_cost
        hold_minutes = max(
            0.0,
            (row.close_time - row.created_time).total_seconds() / 60.0,
        )
        heapq.heappush(
            open_positions,
            (
                row.close_time,
                sequence,
                OpenPosition(
                    ticker=row.ticker,
                    close_time=row.close_time,
                    entry_cost=entry_cost,
                    payout=payout * config.contracts_per_trade,
                    fees=fees,
                    side=side,
                    hold_minutes=hold_minutes,
                ),
            ),
        )
        sequence += 1
        active_tickers.add(row.ticker)
        equity = available_cash + open_cost_basis
        peak_equity, max_drawdown, max_drawdown_pct = update_drawdown(
            equity,
            peak_equity,
            max_drawdown,
            max_drawdown_pct,
        )

    if not frame.empty:
        settle_positions_up_to(frame["close_time"].max())

    ending_equity = available_cash + open_cost_basis
    avg_pnl_per_trade = 0.0 if total_trades == 0 else net_pnl / total_trades
    win_rate = 0.0 if total_trades == 0 else wins / total_trades
    losses = total_trades - wins
    avg_hold_minutes = 0.0 if total_trades == 0 else total_hold_minutes / total_trades
    return_pct = 0.0 if config.starting_capital <= 0 else (ending_equity / config.starting_capital - 1.0) * 100.0

    return {
        "trades": total_trades,
        "yes_trades": yes_trades,
        "no_trades": no_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_pnl_dollars": gross_pnl,
        "total_fees_dollars": total_fees,
        "net_pnl_dollars": net_pnl,
        "avg_pnl_per_trade_dollars": avg_pnl_per_trade,
        "turnover_dollars": turnover,
        "ending_equity_dollars": ending_equity,
        "return_pct": return_pct,
        "peak_equity_dollars": peak_equity,
        "max_drawdown_dollars": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct * 100.0,
        "avg_hold_minutes": avg_hold_minutes,
        "skipped_due_capital": skipped_capital,
        "skipped_due_reserve": skipped_reserve,
        "skipped_due_open_ticker": skipped_stacking,
    }


def format_slug_number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_output_stem(config: StrategyConfig) -> str:
    edge_slug = format_slug_number(config.edge_threshold_cents)
    fee_slug = format_slug_number(config.fee_per_contract_cents)
    slippage_slug = format_slug_number(config.slippage_pct)
    if config.standard_gaussian_entry_mode == "zscore":
        standard_gaussian_slug = f"sgz_{format_slug_number(config.standard_gaussian_z_threshold)}"
    else:
        standard_gaussian_slug = "sg_edge"
    tau_slug = f"tau_{format_slug_number(config.min_tau_minutes)}_to_{format_slug_number(config.max_tau_minutes)}m"
    reserve_slug = f"reserve_{format_slug_number(config.reserve_cash_pct)}pct"
    stack_slug = "stack" if config.allow_stacking else "onepos"
    capital_slug = "capital" if config.enforce_capital_constraint else "nocapital"
    return (
        f"profitability_{config.model_scope}_edge_{edge_slug}c_{standard_gaussian_slug}_fee_{config.fee_mode}_{fee_slug}c_slippage_{slippage_slug}pct_{tau_slug}_{reserve_slug}_"
        f"{config.contracts_per_trade}x_{stack_slug}_{capital_slug}"
    )


def build_markdown_report(
    series_ticker: str,
    metrics_df: pd.DataFrame,
    config: StrategyConfig,
    sample_rows: int,
    test_rows: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament Profitability Evaluation",
        "",
        "- Dataset: held-out tournament test split only",
        f"- Evaluation sample rows: `{sample_rows:,}`",
        f"- Test rows: `{test_rows:,}`",
        f"- Model scope: `{config.model_scope}`",
        "",
        "## Default Assumptions",
        "",
        f"- Entry rule: buy `YES` when `model_prob - market_prob >= {config.edge_threshold_cents:.2f}c`; buy `NO` when the reverse is true",
        (
            f"- Standard Gaussian override: use market `z = norm.ppf(price)` and trade when `|z| >= {config.standard_gaussian_z_threshold:.5f}`"
            if config.standard_gaussian_entry_mode == "zscore"
            else "- Standard Gaussian override: none"
        ),
        "- Exit rule: hold every filled trade to expiry",
        f"- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `{config.slippage_pct:.2f}%` adverse slippage",
        f"- Order size: `{config.contracts_per_trade}` contract(s) per trade",
        f"- Trade window: only rows with `tau_minutes` between `{config.min_tau_minutes:.2f}` and `{config.max_tau_minutes:.2f}` inclusive",
        (
            f"- Fees: `flat {config.fee_per_contract_cents:.2f}c per contract`"
            if config.fee_mode == "flat"
            else "- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent"
        ),
        f"- Position rule: `{'multiple open positions per ticker allowed' if config.allow_stacking else 'one open position per ticker at a time'}`",
        f"- Capital rule: `{'enforce available cash before entry' if config.enforce_capital_constraint else 'do not enforce capital availability'}`",
        f"- Reserve rule: keep at least `{config.reserve_cash_pct:.2f}%` of current portfolio equity uninvested",
        "- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity",
        "",
        "## Results",
        "",
        "| Model | Log-Loss | Trades | Wins | Losses | Win Rate | Total Invested ($) | Net PnL ($) | Return % | Max DD ($) | Max DD % | Skipped Capital | Skipped Reserve |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['log_loss']:.5f} | {int(row['trades']):,} | "
            f"{int(row['wins']):,} | {int(row['losses']):,} | "
            f"{row['win_rate'] * 100.0:.2f}% | "
            f"{row['turnover_dollars']:.2f} | "
            f"{row['net_pnl_dollars']:.2f} | {row['return_pct']:.2f} | "
            f"{row['max_drawdown_dollars']:.2f} | {row['max_drawdown_pct']:.2f} | "
            f"{int(row['skipped_due_capital']):,} | {int(row['skipped_due_reserve']):,} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    artifacts_dir: Path,
    sample_size: int,
    config: StrategyConfig,
) -> None:
    smarterpred = load_smarterpred_module()

    print(f"Series: {series_ticker}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Markets: {markets_path}")
    print(f"Trades: {trades_path}")

    splits = build_profitability_dataset(
        smarterpred=smarterpred,
        series_ticker=series_ticker,
        markets_path=markets_path,
        trades_path=trades_path,
        sample_size=sample_size,
    )
    predictions, overall_log_loss = load_saved_predictions(
        smarterpred=smarterpred,
        artifacts_dir=artifacts_dir,
        splits=splits,
        model_scope=config.model_scope,
    )
    predictions, overall_log_loss = filter_predictions_by_scope(
        predictions=predictions,
        overall_log_loss=overall_log_loss,
        smarterpred=smarterpred,
        model_scope=config.model_scope,
    )
    filtered_test_frame = splits["test_frame"].copy()
    tau_mask = (
        (filtered_test_frame["tau_minutes"] >= config.min_tau_minutes)
        & (filtered_test_frame["tau_minutes"] <= config.max_tau_minutes)
    )
    tau_mask_np = tau_mask.to_numpy()
    filtered_test_frame = filtered_test_frame.loc[tau_mask].reset_index(drop=True)

    metrics_rows = []
    for model_name, model_predictions in predictions.items():
        model_label = smarterpred.MODEL_NAME_TO_LABEL[model_name]
        row = simulate_profitability_for_model(
            model_label=model_label,
            test_frame=filtered_test_frame,
            model_predictions=model_predictions[tau_mask_np],
            config=config,
        )
        row["model"] = model_label
        row["log_loss"] = overall_log_loss[model_label]
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["net_pnl_dollars", "max_drawdown_dollars", "log_loss"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    output_stem = build_output_stem(config)
    output_csv = artifacts_dir / f"{output_stem}.csv"
    output_json = artifacts_dir / f"{output_stem}.json"
    output_md = artifacts_dir / f"{output_stem}.md"

    metrics_df.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(
            {
                "series": series_ticker,
                "config": {
                    "starting_capital": config.starting_capital,
                    "contracts_per_trade": config.contracts_per_trade,
                    "edge_threshold_cents": config.edge_threshold_cents,
                    "standard_gaussian_entry_mode": config.standard_gaussian_entry_mode,
                    "standard_gaussian_z_threshold": config.standard_gaussian_z_threshold,
                    "fee_mode": config.fee_mode,
                    "fee_per_contract_cents": config.fee_per_contract_cents,
                    "slippage_pct": config.slippage_pct,
                    "model_scope": config.model_scope,
                    "min_tau_minutes": config.min_tau_minutes,
                    "max_tau_minutes": config.max_tau_minutes,
                    "reserve_cash_pct": config.reserve_cash_pct,
                    "enforce_capital_constraint": config.enforce_capital_constraint,
                    "allow_stacking": config.allow_stacking,
                },
                "results": metrics_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_md.write_text(
        build_markdown_report(
            series_ticker=series_ticker,
            metrics_df=metrics_df,
            config=config,
            sample_rows=int(splits["sample_rows"]),
            test_rows=int(len(filtered_test_frame)),
        ),
        encoding="utf-8",
    )

    print("\nProfitability results:")
    print(
        metrics_df[
            [
                "model",
                "log_loss",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "turnover_dollars",
                "net_pnl_dollars",
                "return_pct",
                "max_drawdown_dollars",
                "max_drawdown_pct",
                "skipped_due_capital",
                "skipped_due_reserve",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV to {output_csv}")
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tournament models with a separate profitability and drawdown simulation.")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Series ticker prefix, e.g. KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the series markets parquet/csv file")
    parser.add_argument("--trades-path", help="Path to a trades directory or trades parquet file")
    parser.add_argument("--artifacts-dir", help="Directory containing saved tournament artifacts")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional evaluation sample size. Use 0 to evaluate all available rows.",
    )
    parser.add_argument(
        "--model-scope",
        choices=["all", "ml", "lightgbm"],
        default="all",
        help="Which model family to evaluate",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=10000.0,
        help="Starting capital in dollars for the simulator",
    )
    parser.add_argument(
        "--contracts-per-trade",
        type=int,
        default=1,
        help="Fixed number of contracts per qualifying trade",
    )
    parser.add_argument(
        "--edge-threshold-cents",
        type=float,
        default=2.0,
        help="Minimum model-vs-market edge, in cents, required to enter a trade",
    )
    parser.add_argument(
        "--standard-gaussian-entry-mode",
        choices=["edge", "zscore"],
        default="edge",
        help="Entry rule override for the Standard Gaussian baseline",
    )
    parser.add_argument(
        "--standard-gaussian-z-threshold",
        type=float,
        help="Absolute z threshold for Standard Gaussian when --standard-gaussian-entry-mode=zscore",
    )
    parser.add_argument(
        "--fee-mode",
        choices=["flat", "kalshi"],
        default="flat",
        help="Fee model to apply to each trade",
    )
    parser.add_argument(
        "--fee-per-contract-cents",
        type=float,
        default=0.0,
        help="Flat fee charged per contract, in cents. Used only when --fee-mode=flat",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.0,
        help="Adverse slippage applied multiplicatively to entry price, in percent",
    )
    parser.add_argument(
        "--min-tau-minutes",
        type=float,
        default=0.0,
        help="Minimum tau_minutes allowed for trading",
    )
    parser.add_argument(
        "--max-tau-minutes",
        type=float,
        default=15.0,
        help="Maximum tau_minutes allowed for trading",
    )
    parser.add_argument(
        "--reserve-cash-pct",
        type=float,
        default=0.0,
        help="Minimum percent of current portfolio equity that must remain uninvested",
    )
    parser.add_argument(
        "--allow-stacking",
        action="store_true",
        help="Allow multiple overlapping positions in the same ticker",
    )
    parser.add_argument(
        "--disable-capital-constraint",
        action="store_true",
        help="Do not require available cash before entering a trade",
    )
    args = parser.parse_args()

    if args.contracts_per_trade <= 0:
        raise ValueError("--contracts-per-trade must be positive")
    if args.edge_threshold_cents < 0:
        raise ValueError("--edge-threshold-cents must be non-negative")
    if args.standard_gaussian_z_threshold is not None and args.standard_gaussian_z_threshold < 0:
        raise ValueError("--standard-gaussian-z-threshold must be non-negative")
    if args.fee_per_contract_cents < 0:
        raise ValueError("--fee-per-contract-cents must be non-negative")
    if args.slippage_pct < 0:
        raise ValueError("--slippage-pct must be non-negative")
    if args.min_tau_minutes < 0:
        raise ValueError("--min-tau-minutes must be non-negative")
    if args.max_tau_minutes <= 0:
        raise ValueError("--max-tau-minutes must be positive")
    if args.min_tau_minutes > args.max_tau_minutes:
        raise ValueError("--min-tau-minutes must be less than or equal to --max-tau-minutes")
    if args.reserve_cash_pct < 0 or args.reserve_cash_pct >= 100:
        raise ValueError("--reserve-cash-pct must be between 0 and 100")
    if args.starting_capital <= 0:
        raise ValueError("--starting-capital must be positive")

    smarterpred = load_smarterpred_module()
    markets_path, trades_path = smarterpred.resolve_inputs(args.series, args.markets_path, args.trades_path)
    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir, args.series)
    standard_gaussian_z_threshold = args.standard_gaussian_z_threshold
    if standard_gaussian_z_threshold is None:
        standard_gaussian_z_threshold = float(norm.ppf(min(0.99, 0.5 + (args.edge_threshold_cents / 100.0))))

    config = StrategyConfig(
        starting_capital=args.starting_capital,
        contracts_per_trade=args.contracts_per_trade,
        edge_threshold_cents=args.edge_threshold_cents,
        standard_gaussian_entry_mode=args.standard_gaussian_entry_mode,
        standard_gaussian_z_threshold=standard_gaussian_z_threshold,
        fee_mode=args.fee_mode,
        fee_per_contract_cents=args.fee_per_contract_cents,
        slippage_pct=args.slippage_pct,
        model_scope=args.model_scope,
        min_tau_minutes=args.min_tau_minutes,
        max_tau_minutes=args.max_tau_minutes,
        reserve_cash_pct=args.reserve_cash_pct,
        enforce_capital_constraint=not args.disable_capital_constraint,
        allow_stacking=args.allow_stacking,
    )

    run_evaluation(
        series_ticker=args.series,
        markets_path=markets_path,
        trades_path=trades_path,
        artifacts_dir=artifacts_dir,
        sample_size=args.sample_size,
        config=config,
    )


if __name__ == "__main__":
    main()
