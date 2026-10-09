# Allocation policy sensitivity

Same CP-SAT function and hard rules for each run; only integer benefit weights change.

Own-policy points have different scales. Compare balanced-reference points across rows.

| Policy | Served | Chilled delivered (m3) | Own points | Common balanced points | Deferred | Status |
|---|---:|---:|---:|---:|---:|---|
| Fresh-first | 79 | 140.723 | 100300 | 35770 | 6 | OPTIMAL |
| Fairness-first | 77 | 126.254 | 60200 | 34900 | 8 | OPTIMAL |
| Balanced | 79 | 140.723 | 35770 | 35770 | 6 | OPTIMAL |

The policies change the served/deferred decision for 6 orders. The order-level differences are recorded in policy_changed_orders.csv.

The verified Balanced allocation remains the submission. Every candidate passes the organizer checker. The cold-chain demand of 181.629 m3 exceeds the optimistic two-trip refrigeration capacity of 172.400 m3 by 9.229 m3. That simple bound already rules out serving all chilled volume. Whole-order packing, district separation, weight, van access and time impose additional limits. The 79-order maximum-count proof is reported separately in task2b_summary.json. Optimality covers the published competition model.
