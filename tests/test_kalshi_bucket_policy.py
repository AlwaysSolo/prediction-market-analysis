from __future__ import annotations

from src.live.kalshi.bucket_policy import (
    DEFAULT_BANNED_NO_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
    DEFAULT_BANNED_YES_TAU_BUCKETS,
    build_chosen_side_buckets,
    evaluate_bucket_ban_policy,
    label_edge_bucket,
    label_price_bucket,
    label_probability_bucket,
    label_tau_bucket,
    parse_bucket_csv,
)


def test_bucket_labels_respect_expected_boundaries() -> None:
    assert label_tau_bucket(2.0) == "2-4"
    assert label_tau_bucket(14.0) == "12-14"
    assert label_price_bucket(0) == "0-10"
    assert label_price_bucket(100) == "90-100"
    assert label_probability_bucket(0.40) == "40-50"
    assert label_probability_bucket(1.0) == "90-100"
    assert label_edge_bucket(-1.0) == "<0"
    assert label_edge_bucket(4.99) == "0-5"
    assert label_edge_bucket(60.0) == "60+"


def test_build_chosen_side_buckets_uses_side_facing_probability() -> None:
    yes_buckets = build_chosen_side_buckets(
        side="YES",
        tau_minutes=3.5,
        entry_price_cents=38,
        predicted_yes_probability=0.44,
        chosen_edge_cents=3.0,
    )
    no_buckets = build_chosen_side_buckets(
        side="NO",
        tau_minutes=3.5,
        entry_price_cents=28,
        predicted_yes_probability=0.44,
        chosen_edge_cents=11.0,
    )

    assert yes_buckets.chosen_side_probability_bucket == "40-50"
    assert no_buckets.chosen_side_probability_bucket == "50-60"


def test_parse_bucket_csv_uses_defaults_for_empty_values() -> None:
    assert parse_bucket_csv(None, default=DEFAULT_BANNED_YES_TAU_BUCKETS) == DEFAULT_BANNED_YES_TAU_BUCKETS
    assert parse_bucket_csv("", default=DEFAULT_BANNED_NO_PRICE_BUCKETS) == DEFAULT_BANNED_NO_PRICE_BUCKETS
    assert parse_bucket_csv("0-10, 10-20", default=DEFAULT_BANNED_YES_PRICE_BUCKETS) == frozenset({"0-10", "10-20"})


def test_evaluate_bucket_ban_policy_blocks_expected_buckets() -> None:
    yes_tau_block = evaluate_bucket_ban_policy(
        enabled=True,
        side="YES",
        buckets=build_chosen_side_buckets(
            side="YES",
            tau_minutes=2.5,
            entry_price_cents=55,
            predicted_yes_probability=0.72,
            chosen_edge_cents=8.0,
        ),
        banned_yes_tau_buckets=DEFAULT_BANNED_YES_TAU_BUCKETS,
        banned_yes_price_buckets=DEFAULT_BANNED_YES_PRICE_BUCKETS,
        banned_yes_probability_buckets=DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
        banned_no_price_buckets=DEFAULT_BANNED_NO_PRICE_BUCKETS,
    )
    no_price_block = evaluate_bucket_ban_policy(
        enabled=True,
        side="NO",
        buckets=build_chosen_side_buckets(
            side="NO",
            tau_minutes=9.0,
            entry_price_cents=28,
            predicted_yes_probability=0.65,
            chosen_edge_cents=12.0,
        ),
        banned_yes_tau_buckets=DEFAULT_BANNED_YES_TAU_BUCKETS,
        banned_yes_price_buckets=DEFAULT_BANNED_YES_PRICE_BUCKETS,
        banned_yes_probability_buckets=DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
        banned_no_price_buckets=DEFAULT_BANNED_NO_PRICE_BUCKETS,
    )
    allowed = evaluate_bucket_ban_policy(
        enabled=True,
        side="YES",
        buckets=build_chosen_side_buckets(
            side="YES",
            tau_minutes=8.5,
            entry_price_cents=65,
            predicted_yes_probability=0.74,
            chosen_edge_cents=9.0,
        ),
        banned_yes_tau_buckets=DEFAULT_BANNED_YES_TAU_BUCKETS,
        banned_yes_price_buckets=DEFAULT_BANNED_YES_PRICE_BUCKETS,
        banned_yes_probability_buckets=DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
        banned_no_price_buckets=DEFAULT_BANNED_NO_PRICE_BUCKETS,
    )

    assert yes_tau_block.is_blocked is True
    assert yes_tau_block.blocked_dimension == "tau"
    assert no_price_block.is_blocked is True
    assert no_price_block.blocked_dimension == "price"
    assert allowed.is_blocked is False
