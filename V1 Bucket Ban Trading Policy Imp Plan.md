# V1 Bucket-Ban Trading Policy

## Summary
Implement a new **bucket-ban policy layer** that blocks structurally weak trades in both the **live execution path** and the **research sampler path**. The policy should evaluate the **chosen side’s** tau, price, and chosen-side probability buckets and block a trade if it matches any banned bucket for that side.

Chosen defaults:
- Apply in **both** live execution and research
- Ship **enabled by default**, but **runtime-configurable via env/config**
- Use the current **Aggressive V1** ban set derived from the live production regime run

Default ban set:
- Ban `YES` when `tau_bucket == "2-4"`
- Ban `YES` when `price_bucket` is one of: `0-10`, `10-20`, `20-30`, `30-40`
- Ban `YES` when `chosen_side_probability_bucket` is one of: `0-10`, `10-20`, `20-30`, `30-40`, `40-50`
- Ban `NO` when `price_bucket == "20-30"`

Rationale for this default:
- On the current production regime run, removing those weak YES structures from settled trades would have improved settled PnL from about `+$595.57` to about `+$794.15` on a direct subtraction counterfactual.
- Edge buckets are **not** part of v1; the weakness is concentrated in **YES tau / price / probability structure**, not edge.

## Key Changes
### 1. Shared bucket-policy utility
Add one shared utility module for:
- bucket labeling:
  - tau
  - price
  - chosen-side probability
  - edge
- bucket-ban evaluation on the **chosen side**
- returning a structured result such as:
  - `is_blocked`
  - `blocked_dimension` (`tau`, `price`, `probability`)
  - `blocked_bucket`
  - `blocked_side`

This should become the single source of truth for both:
- `src/live/kalshi/signal_risk.py`
- `src/live/kalshi/research/engine.py`

Do not leave duplicated bucket-label code in multiple places after this change.

### 2. Config and env interface
Extend both config dataclasses with a bucket-ban section:
- `enable_bucket_ban_policy: bool`
- `banned_yes_tau_buckets: set[str]`
- `banned_yes_price_buckets: set[str]`
- `banned_yes_probability_buckets: set[str]`
- `banned_no_price_buckets: set[str]`

Load them from env using the existing path-specific patterns:
- signal path:
  - `KALSHI_[DEMO|PROD]_SIGNAL_ENABLE_BUCKET_BAN_POLICY`
  - `KALSHI_[DEMO|PROD]_SIGNAL_BANNED_YES_TAU_BUCKETS`
  - `KALSHI_[DEMO|PROD]_SIGNAL_BANNED_YES_PRICE_BUCKETS`
  - `KALSHI_[DEMO|PROD]_SIGNAL_BANNED_YES_PROBABILITY_BUCKETS`
  - `KALSHI_[DEMO|PROD]_SIGNAL_BANNED_NO_PRICE_BUCKETS`
- research path:
  - `KALSHI_[DEMO|PROD]_RESEARCH_ENABLE_BUCKET_BAN_POLICY`
  - `KALSHI_[DEMO|PROD]_RESEARCH_BANNED_YES_TAU_BUCKETS`
  - `KALSHI_[DEMO|PROD]_RESEARCH_BANNED_YES_PRICE_BUCKETS`
  - `KALSHI_[DEMO|PROD]_RESEARCH_BANNED_YES_PROBABILITY_BUCKETS`
  - `KALSHI_[DEMO|PROD]_RESEARCH_BANNED_NO_PRICE_BUCKETS`

Use comma-separated bucket labels in env values.
Default those env-backed fields to the Aggressive V1 ban set above.

### 3. Live execution integration
In `signal_risk`, evaluate bucket policy:
- after a side has been chosen
- after chosen-side bucket labels are available
- before regime hard gate
- before stacking / ticker lock / cooldown / cash checks

If blocked:
- return a blocked candidate with `block_reason = "blocked_by_bucket_policy"`
- include the evaluated bucket labels in the decision/update payload
- include bucket-policy detail fields:
  - `bucket_policy_dimension`
  - `bucket_policy_bucket`
  - `bucket_policy_side`

Precedence in live path:
1. no viable candidate / existing rejection reasons
2. bucket-ban policy
3. regime hard gate
4. ticker lock / cooldown / cash / reserve

### 4. Research sampler integration
In the research sampler, evaluate bucket policy:
- after chosen-side evaluation
- after computing the chosen-side bucket labels
- after `duplicate_signature`
- before regime hard gate
- before sample recording

If blocked:
- publish `research_sample_skipped`
- use `reason = "blocked_by_bucket_policy"`
- include:
  - chosen-side bucket labels
  - `bucket_policy_dimension`
  - `bucket_policy_bucket`
  - `bucket_policy_side`

Keep current signature behavior unchanged:
- blocked signatures are **not** marked as seen
- repeated blocked observations may still appear in skip logs, consistent with current regime-block behavior

Precedence in research path:
1. no viable candidate / existing rejection reason
2. `duplicate_signature`
3. bucket-ban policy
4. regime hard gate
5. record sample

### 5. Logging and analysis compatibility
Extend serialized blocked/decision payloads so later analysis can distinguish:
- ordinary skip/block reason
- bucket-policy side
- bucket-policy dimension
- bucket-policy bucket
- the full chosen-side bucket labels

This should flow through existing performance analysis without needing a new analyzer mode. At minimum, the new block reason and fields must be preserved in canonical rows / skip summaries wherever blocked decisions are already surfaced.

## Test Plan
- Bucket-label helper tests:
  - exact boundary behavior for tau, price, probability, and edge labels
  - chosen-side probability labeling uses YES probability for YES and `1 - predicted_yes_probability` for NO
- Config/env parsing tests:
  - comma-separated bucket lists parse into sets
  - defaults match the Aggressive V1 set
  - empty env values fall back cleanly
- Live policy tests:
  - `YES` with `tau 2-4` is blocked
  - `YES` with `price 30-40` is blocked
  - `YES` with `probability 40-50` is blocked
  - `NO` with `price 20-30` is blocked
  - allowed buckets still pass through to regime / lock / cash logic
  - blocked payload includes `blocked_by_bucket_policy` and bucket detail fields
- Research sampler tests:
  - same side/bucket cases above are skipped with `blocked_by_bucket_policy`
  - allowed candidates still record normally
  - `duplicate_signature` still wins over bucket policy when the signature is already seen
- Integration tests:
  - research run with policy enabled still writes normal research logs plus the new block reason/details
  - live path still emits valid decision updates and does not regress existing regime behavior

## Assumptions and Defaults
- v1 uses **broad single-dimension bans**, not combo-level bans
- v1 applies to **both** live execution and research
- v1 is **enabled by default** through config defaults, but fully overridable by env
- v1 blocks based on the **chosen side’s** buckets only
- v1 does **not** ban any edge buckets
- v1 uses one block reason, `blocked_by_bucket_policy`, plus separate detail fields instead of many specialized reason strings
- No model retraining or artifact format changes are part of this work
