from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.kalshi import KalshiEnvironment, KalshiExecutionConfig, KalshiExecutionMode
from scripts.run_kalshi_bagged_lasso_live import (
    DEDICATED_LIVE_SIGNAL_PROFILE,
    RESEARCH_PARITY_SIGNAL_PROFILE,
    feature_schema_for_model_file,
    requires_external_spot_feature_schema,
    _build_live_signal_config,
    validate_configured_subaccount_number,
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

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        signal_profile=DEDICATED_LIVE_SIGNAL_PROFILE,
    )

    assert config.edge_threshold_cents == 6.0
    assert config.min_tau_minutes == 0.0
    assert config.max_tau_minutes == 10.0
    assert config.price_band_min_cents == 10
    assert config.price_band_max_cents == 90
    assert config.reserve_cash_pct == 30.0
    assert config.slippage_pct == 1.0
    assert config.apply_regime_hard_gate is True
    assert config.enable_bucket_ban_policy is True
    assert config.enable_combo_ban_policy is True
    assert config.contracts_per_order == 1
    assert config.capital_pct_per_order is None
    assert config.kelly_fraction_multiplier is None
    assert config.kelly_fraction_cap_pct is None
    assert config.max_ticker_side_exposure_dollars == 1.0
    assert config.max_tau_minutes == 10.0
    assert config.blocked_regime_labels == frozenset({"neutral"})
    assert config.banned_combo_buckets == frozenset(
        {
            "12-14|60-70|70-80|5-10",
            "10-12|40-50|50-60|5-10",
            "10-12|30-40|30-40|5-10",
        }
    )
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
        signal_profile=DEDICATED_LIVE_SIGNAL_PROFILE,
        allow_stacking=True,
    )

    assert config.allow_stacking is True


def test_build_live_signal_config_supports_research_parity_profile(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 6.0,
                    "min_tau_minutes": 1.0,
                    "max_tau_minutes": 12.0,
                    "price_band_min_cents": 10,
                    "price_band_max_cents": 90,
                    "allow_stacking": False,
                    "enable_combo_ban_policy": True,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        signal_profile=RESEARCH_PARITY_SIGNAL_PROFILE,
        allow_stacking=True,
    )

    assert config.edge_threshold_cents == 2.0
    assert config.min_tau_minutes == 0.0
    assert config.max_tau_minutes == 15.0
    assert config.price_band_min_cents == 0
    assert config.price_band_max_cents == 100
    assert config.apply_regime_hard_gate is True
    assert config.enable_bucket_ban_policy is True
    assert config.enable_combo_ban_policy is False
    assert config.banned_combo_buckets == frozenset()
    assert config.blocked_regime_labels == frozenset()
    assert config.contracts_per_order == 1
    assert config.capital_pct_per_order is None
    assert config.kelly_fraction_multiplier is None
    assert config.kelly_fraction_cap_pct is None
    assert config.max_ticker_side_exposure_dollars == 1.0
    assert config.allow_stacking is True


def test_build_live_signal_config_can_disable_regime_control(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 6.0,
                    "allow_stacking": False,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        signal_profile=DEDICATED_LIVE_SIGNAL_PROFILE,
        disable_regime_control=True,
    )

    assert config.apply_regime_hard_gate is False
    assert config.blocked_regime_labels == frozenset()
    assert config.enable_combo_ban_policy is True


def test_build_live_signal_config_can_disable_bucket_controls(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 6.0,
                    "price_band_min_cents": 20,
                    "price_band_max_cents": 80,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        signal_profile=DEDICATED_LIVE_SIGNAL_PROFILE,
        disable_bucket_controls=True,
    )

    assert config.price_band_min_cents == 0
    assert config.price_band_max_cents == 100
    assert config.enable_bucket_ban_policy is False
    assert config.enable_combo_ban_policy is False
    assert config.banned_combo_buckets == frozenset()
    assert config.blocked_regime_labels == frozenset({"neutral"})


def test_build_live_signal_config_supports_explicit_edge_threshold_override(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 6.0,
                }
            }
        ),
        encoding="utf-8",
    )

    config = _build_live_signal_config(
        KalshiEnvironment.PRODUCTION,
        policy_path,
        signal_profile=DEDICATED_LIVE_SIGNAL_PROFILE,
        edge_threshold_cents=2.0,
    )

    assert config.edge_threshold_cents == 2.0


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


def test_validate_live_runner_preflight_allows_primary_account_subaccount_zero() -> None:
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


def test_validate_configured_subaccount_number_accepts_known_subaccount() -> None:
    validate_configured_subaccount_number(
        configured_subaccount=7,
        subaccount_balances=[
            {"subaccount_number": 0, "balance": 100},
            {"subaccount_number": 7, "balance": 2500},
        ],
    )


def test_validate_configured_subaccount_number_rejects_unknown_subaccount() -> None:
    with pytest.raises(RuntimeError, match="Valid subaccount numbers: 0, 7"):
        validate_configured_subaccount_number(
            configured_subaccount=123,
            subaccount_balances=[
                {"subaccount_number": 0, "balance": 100},
                {"subaccount_number": 7, "balance": 2500},
            ],
        )


def test_validate_configured_subaccount_number_rejects_empty_response() -> None:
    with pytest.raises(RuntimeError, match="did not return any subaccount numbers"):
        validate_configured_subaccount_number(
            configured_subaccount=7,
            subaccount_balances=[],
        )


def test_feature_schema_for_model_file_reads_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "bagged_lasso").mkdir(parents=True)
    model_file = run_dir / "bagged_lasso" / "model.joblib"
    model_file.write_text("placeholder", encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text(
        json.dumps({"schema_name": "spot_v1"}),
        encoding="utf-8",
    )

    assert feature_schema_for_model_file(model_file) == "spot_v1"


def test_requires_external_spot_feature_schema_only_for_spot_v1() -> None:
    assert requires_external_spot_feature_schema("spot_v1") is True
    assert requires_external_spot_feature_schema("default") is False
    assert requires_external_spot_feature_schema(None) is False
