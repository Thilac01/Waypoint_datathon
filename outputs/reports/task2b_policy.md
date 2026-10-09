# Peak-day prioritization policy

Serve Fresh early, protect the cold chain, and increase priority for repeat deferrals. The transparent benefit per whole order is 100 base points + the brand weight (Fresh 300, Style 0, Tech 0) + 150 for chilled goods + 200 if deferred yesterday + 15 per day since service, capped at 7 days. These are our policy choices, not organizer scoring weights. Among equal priority totals, prefer more orders, then volume. Change `priority_config.json` to examine another policy.

There are 28 available vehicles, including 4 refrigerated vehicles (1 refrigerated van). Chilled demand is 181.629 m3, including 6.013 m3 at van-only outlets. Twice the available refrigerated volume is 172.400 m3: an optimistic upper bound before district separation, weight, whole-order packing and time. Fleet-wide volume is not interchangeable with refrigerated or van-access capacity.

The selected plan serves 79 of 85 orders, delivers 328.298 m3 and defers 6 orders / 81.566 m3. Solver status: OPTIMAL; objective gap 0.000000%. A separate maximum-order-count run serves 79 orders (status OPTIMAL). The priority plan scores 35770 policy points versus 35770 for the count plan. Optimality, when proven, is only for the stated objective and published model.

Every trip has one brand and district, respects both capacities, depot, refrigeration and van restrictions, and each vehicle has at most two trips. Duration = outbound + (orders - 1) x inter-stop + sum(brand/dock allowances). Fresh trip durations sum to at most 270 minutes per vehicle; Style/Tech durations sum to at most 480 minutes separately. No return leg is added. The detailed calculations are in `task2b_trip_audit.csv`.

Individually infeasible orders: S1-078. Other deferrals are choices under shared resources, not proven individually unavoidable. Their order-level reasons and opportunity cost (volume and policy points) are in `task2b_order_decisions.csv`. This is the published Task2B planning model: it does not claim a stop-by-stop route, weekly fuel ledger, or exact arrival-window feasibility beyond the supplied daily budgets.
