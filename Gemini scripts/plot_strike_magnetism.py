from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = Path(__file__).with_name("magnetism_penalty_matrix.csv")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("magnetism_penalty_heatmap.png")


def plot_heatmap(input_path: Path, output_path: Path, min_trade_count: int | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Magnetism matrix not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Magnetism matrix is empty.")

    if min_trade_count is not None and "trade_count" in df.columns:
        df = df[df["trade_count"] >= min_trade_count].copy()
        if df.empty:
            raise ValueError("No rows left after applying the trade_count filter.")

    pivot = df.pivot(index="tau_bin", columns="distance_bin", values="magnetism_penalty").sort_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    vmax = float(df["magnetism_penalty"].abs().max())
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_title("Kalshi Strike Magnetism Penalty Heatmap")
    ax.set_xlabel("Distance From Midpoint (cents)")
    ax.set_ylabel("Time To Close Bucket (minutes)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{col:.0f}" for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{idx:.0f}" for idx in pivot.index])

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Magnetism Penalty = Implied Prob - Actual Win Rate")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a heatmap from the strike magnetism penalty matrix.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Path to magnetism_penalty_matrix.csv")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output path for the heatmap image")
    parser.add_argument(
        "--min-trade-count",
        type=int,
        help="Optional additional filter applied before plotting",
    )
    args = parser.parse_args()

    plot_heatmap(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        min_trade_count=args.min_trade_count,
    )


if __name__ == "__main__":
    main()
