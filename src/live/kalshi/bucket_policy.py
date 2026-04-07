from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

TAU_BUCKETS: tuple[tuple[float, float], ...] = (
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, 12.0),
    (12.0, 14.0),
)
PRICE_BUCKETS: tuple[tuple[int, int], ...] = tuple((value, value + 10) for value in range(0, 100, 10))
PROBABILITY_BUCKETS: tuple[tuple[float, float], ...] = tuple((value / 100.0, (value + 10) / 100.0) for value in range(0, 100, 10))
EDGE_BUCKETS: tuple[tuple[float, float | None], ...] = (
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, None),
)

DEFAULT_BANNED_YES_TAU_BUCKETS: frozenset[str] = frozenset({"2-4"})
DEFAULT_BANNED_YES_PRICE_BUCKETS: frozenset[str] = frozenset({"0-10", "10-20", "20-30", "30-40"})
DEFAULT_BANNED_YES_PROBABILITY_BUCKETS: frozenset[str] = frozenset({"0-10", "10-20", "20-30", "30-40", "40-50"})
DEFAULT_BANNED_NO_PRICE_BUCKETS: frozenset[str] = frozenset({"20-30"})


@dataclass(frozen=True)
class KalshiChosenSideBuckets:
    tau_bucket: str
    price_bucket: str
    chosen_side_probability_bucket: str
    chosen_side_edge_bucket: str


@dataclass(frozen=True)
class KalshiBucketPolicyEvaluation:
    is_blocked: bool
    blocked_dimension: str | None = None
    blocked_bucket: str | None = None
    blocked_side: str | None = None


def parse_bucket_csv(value: str | None, *, default: Collection[str]) -> frozenset[str]:
    if value is None:
        return frozenset(default)
    parsed = frozenset(part.strip() for part in value.split(",") if part.strip())
    return frozenset(default) if not parsed else parsed


def label_price_bucket(price_cents: int | None) -> str:
    if price_cents is None:
        return "unknown"
    for lower, upper in PRICE_BUCKETS:
        if lower <= price_cents < upper:
            return f"{lower}-{upper}"
    if price_cents == 100:
        return "90-100"
    return "unknown"


def label_probability_bucket(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    clamped = max(0.0, min(1.0, float(probability)))
    for lower, upper in PROBABILITY_BUCKETS:
        if lower <= clamped < upper:
            return f"{int(lower * 100)}-{int(upper * 100)}"
    return "90-100"


def label_tau_bucket(tau_minutes: float | None) -> str:
    if tau_minutes is None:
        return "unknown"
    for lower, upper in TAU_BUCKETS:
        if lower <= tau_minutes < upper or (upper == TAU_BUCKETS[-1][1] and tau_minutes <= upper):
            return f"{int(lower)}-{int(upper)}"
    return "unknown"


def label_edge_bucket(edge_cents: float | None) -> str:
    if edge_cents is None:
        return "unknown"
    if edge_cents < 0.0:
        return "<0"
    for lower, upper in EDGE_BUCKETS:
        if upper is None and edge_cents >= lower:
            return f"{int(lower)}+"
        if upper is not None and lower <= edge_cents < upper:
            return f"{int(lower)}-{int(upper)}"
    return "unknown"


def chosen_side_probability(side: str | None, predicted_yes_probability: float | None) -> float | None:
    if side not in {"YES", "NO"} or predicted_yes_probability is None:
        return None
    if side == "YES":
        return predicted_yes_probability
    return 1.0 - predicted_yes_probability


def build_chosen_side_buckets(
    *,
    side: str | None,
    tau_minutes: float | None,
    entry_price_cents: int | None,
    predicted_yes_probability: float | None,
    chosen_edge_cents: float | None,
) -> KalshiChosenSideBuckets:
    return KalshiChosenSideBuckets(
        tau_bucket=label_tau_bucket(tau_minutes),
        price_bucket=label_price_bucket(entry_price_cents),
        chosen_side_probability_bucket=label_probability_bucket(chosen_side_probability(side, predicted_yes_probability)),
        chosen_side_edge_bucket=label_edge_bucket(chosen_edge_cents),
    )


def evaluate_bucket_ban_policy(
    *,
    enabled: bool,
    side: str | None,
    buckets: KalshiChosenSideBuckets,
    banned_yes_tau_buckets: Collection[str],
    banned_yes_price_buckets: Collection[str],
    banned_yes_probability_buckets: Collection[str],
    banned_no_price_buckets: Collection[str],
) -> KalshiBucketPolicyEvaluation:
    if not enabled or side not in {"YES", "NO"}:
        return KalshiBucketPolicyEvaluation(is_blocked=False)

    if side == "YES":
        if buckets.tau_bucket in banned_yes_tau_buckets:
            return KalshiBucketPolicyEvaluation(True, "tau", buckets.tau_bucket, side)
        if buckets.price_bucket in banned_yes_price_buckets:
            return KalshiBucketPolicyEvaluation(True, "price", buckets.price_bucket, side)
        if buckets.chosen_side_probability_bucket in banned_yes_probability_buckets:
            return KalshiBucketPolicyEvaluation(True, "probability", buckets.chosen_side_probability_bucket, side)
        return KalshiBucketPolicyEvaluation(is_blocked=False)

    if buckets.price_bucket in banned_no_price_buckets:
        return KalshiBucketPolicyEvaluation(True, "price", buckets.price_bucket, side)
    return KalshiBucketPolicyEvaluation(is_blocked=False)


def bucket_policy_fields(evaluation: KalshiBucketPolicyEvaluation | None) -> dict[str, str | None]:
    if evaluation is None:
        return {
            "bucket_policy_dimension": None,
            "bucket_policy_bucket": None,
            "bucket_policy_side": None,
        }
    return {
        "bucket_policy_dimension": evaluation.blocked_dimension,
        "bucket_policy_bucket": evaluation.blocked_bucket,
        "bucket_policy_side": evaluation.blocked_side,
    }
