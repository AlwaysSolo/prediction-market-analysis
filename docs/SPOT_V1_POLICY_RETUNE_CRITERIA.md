# spot_v1 Policy Retune Criteria

This file records the decision rules before looking at any retuned threshold results.

## Scope

These criteria apply to the March 2026 pilot retune on the purged common-row comparison set for:

- `default_pilot_march_walkforward_purged_regime_diag`
- `spot_v1_pilot_march_walkforward_purged_regime_diag`

The March pilot is a policy-tuning checkpoint only. It is not sufficient by itself for production promotion.

## Carry-Forward Criteria

The retuned `spot_v1` policy is good enough to carry forward to the next out-of-sample month only if all of the following are true on the held-out evaluation half:

1. `net_pnl_dollars` is strictly better than the untuned `spot_v1` policy.
2. `net_pnl_dollars` is strictly better than the `default` baseline policy.
3. Conditional log loss does not regress by more than `0.005` on any of these slices versus the untuned `spot_v1` policy:
   - default-selected rows
   - spot-selected rows
   - default near-threshold rows
   - spot near-threshold rows
4. Medium-volatility evaluation PnL improves by at least `$1.00` versus the untuned `spot_v1` policy.
5. High-volatility evaluation PnL does not worsen by more than `$1.00` versus the untuned `spot_v1` policy.
6. Tail calibration confirmation does not show a worse absolute calibration error by more than `0.02` in either tail bucket:
   - calibrated probability `< 0.10`
   - calibrated probability `> 0.90`

If any one of the above fails, the retune is not considered good enough to carry forward.

## Promotion Criteria

No March-only retune result is sufficient for production promotion.

Promotion requires the locked retuned policy to beat both the untuned `spot_v1` policy and the `default` policy on the next out-of-sample month, while preserving the carry-forward criteria above.
