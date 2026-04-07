from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.kalshi import KalshiEnvironment, KalshiExecutionConfig, KalshiExecutionMode
from scripts.run_kalshi_bagged_lasso_live import (
    _build_live_signal_config,
    validate_live_runner_preflight,
)


def test_build_live_signal_config_forces_live_profile_overrides(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 6.0,
                    "min_tau_minutes": 0.0,
                    "max_tau_minutes": 12.0,
                    "price_band_min_cents": 10,
                    "price_band_max_cents": 90,
                    "reserve_cash_pct": 30.0,
                    "slippage_pct": 1.0,
                    "allow_stacking": True,
                    "contracts_per_order": 3,
                    "capital_pct_per_order": 5.0,
                    "kelly_fraction_multiplier": 0.5,
                    "kelly_fraction_cap_pct": 2.0,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(KalshiEnvironment.PRODUCTION, policy_path)

    assert config.edge_threshold_cents == 6.0
    assert config.min_tau_minutes == 0.0
    assert config.max_tau_minutes == 12.0
    assert config.price_band_min_cents == 10
    assert config.price_band_max_cents == 90
    assert config.reserve_cash_pct == 30.0
    assert config.slippage_pct == 1.0
    assert config.apply_regime_hard_gate is True
    assert config.enable_bucket_ban_policy is True
    assert config.contracts_per_order == 1
    assert config.capital_pct_per_order is None
    assert config.kelly_fraction_multiplier is None
    assert config.kelly_fraction_cap_pct is None
    assert config.allow_stacking is False


def test_build_live_signal_config_allows_explicit_stacking_override(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "allow_stacking": False,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        allow_stacking=True,
    )

    assert config.allow_stacking is True


def test_validate_live_runner_preflight_rejects_missing_confirm_live() -> None:
    with pytest.raises(RuntimeError, match="--confirm-live"):
        validate_live_runner_preflight(
            mode="live",
            confirm_live=False,
            execution_config=KalshiExecutionConfig(
                mode=KalshiExecutionMode.LIVE,
                enable_live_trading=True,
                subaccount=7,
            ),
        )


def test_validate_live_runner_preflight_rejects_zero_subaccount() -> None:
    with pytest.raises(RuntimeError, match="nonzero production subaccount"):
        validate_live_runner_preflight(
            mode="shadow",
            confirm_live=False,
            execution_config=KalshiExecutionConfig(
                mode=KalshiExecutionMode.SHADOW,
                enable_live_trading=False,
                subaccount=0,
            ),
        )


def test_validate_live_runner_preflight_rejects_live_without_enable_flag() -> None:
    with pytest.raises(RuntimeError, match="ENABLE_LIVE_TRADING"):
        validate_live_runner_preflight(
            mode="live",
            confirm_live=True,
            execution_config=KalshiExecutionConfig(
                mode=KalshiExecutionMode.LIVE,
                enable_live_trading=False,
                subaccount=7,
            ),
        )


def test_validate_live_runner_preflight_allows_shadow_mode() -> None:
    validate_live_runner_preflight(
        mode="shadow",
        confirm_live=False,
        execution_config=KalshiExecutionConfig(
            mode=KalshiExecutionMode.SHADOW,
            enable_live_trading=False,
            subaccount=7,
        ),
    )
