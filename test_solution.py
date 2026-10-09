"""Focused checks for labels, leakage, order counts and published constraints.

Run: python test_solution.py --data data --output outputs
The original competition data is required; this suite does not download it.
"""
import argparse
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import waypoint_datathon as w


class ChallengeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = w.Data(ARGS.data)
        cls.x, cls.y, cls.late, cls.meta = w.construct_task1(cls.data, True)

    def test_waiting_is_not_service_and_close_is_not_late(self):
        # Actual sample: 04:48 arrival, 05:00 opening, 05:12 departure means 12 min.
        orders = self.data.read('deliveries_train.csv')
        legs = self.data.read('route_legs_train.csv')
        sample = legs[(legs.arrival_time == '04:48') & (legs.leave_outlet_time == '05:12')].iloc[0]
        order = orders[(orders.route_id == sample.route_id) & (orders.seq_in_route == sample.seq)].iloc[0]
        self.assertEqual(order.window_open_time, '05:00')
        i = self.meta.index[self.meta.delivery_id.eq(order.delivery_id)][0]
        self.assertEqual(self.y[i], 12)
        self.assertEqual(self.late[i], 0)
        merged = orders.dropna(subset=['route_id']).merge(legs, left_on=['route_id','seq_in_route'], right_on=['route_id','seq'])
        equal_ids = merged.loc[merged.arrival_time.eq(merged.window_close_time), 'delivery_id']
        self.assertGreater(len(equal_ids), 0)
        self.assertTrue((self.late[self.meta.delivery_id.isin(equal_ids)] == 0).all())

    def test_actual_outcomes_cannot_change_features(self):
        original = self.data.cache['route_legs_train.csv'].copy()
        try:
            changed = original.copy()
            # Add five minutes to every completion; labels must change, features cannot.
            minute = w.minutes(changed.leave_outlet_time).astype(int) + 5
            changed['leave_outlet_time'] = (minute // 60).astype(str).str.zfill(2) + ':' + (minute % 60).astype(str).str.zfill(2)
            self.data.cache['route_legs_train.csv'] = changed
            x2, y2, _, _ = w.construct_task1(self.data, True)
            pd.testing.assert_frame_equal(self.x, x2)
            np.testing.assert_allclose(y2 - self.y, 5)
        finally:
            self.data.cache['route_legs_train.csv'] = original

    def test_demand_includes_not_run_and_uses_order_date(self):
        history, future, keys, report = w.demand_panel(self.data)
        orders = pd.concat([self.data.read('deliveries_train.csv'), self.data.read('task1_test_inputs.csv')])
        self.assertEqual(report['orders_counted'], len(orders))
        self.assertGreater(report['not_run_orders_counted'], 0)
        self.assertAlmostEqual(history.total.sum(), orders.order_volume_m3.sum(), places=6)
        day = orders.loc[orders.dispatch_status.eq('deferred') & orders.order_date.ne(orders.dispatch_date), 'order_date'].iloc[0]
        self.assertAlmostEqual(history.loc[history.date.eq(pd.Timestamp(day)), 'total'].sum(),
                               orders.loc[orders.order_date.eq(day), 'order_volume_m3'].sum(), places=6)
        self.assertEqual(keys.week_start.nunique(), 10)

    def test_submissions_and_corruption_detection(self):
        out = Path(ARGS.output)
        if not (out/'submissions/submission_task1.csv').exists():
            self.skipTest('Run all tasks first to validate saved submissions')
        w.validate_submissions(self.data, out)
        sub = pd.read_csv(out/'submissions/submission_task2b.csv')
        changed = sub.copy()
        workshop = self.data.read('task2b_peak_day_fleet.csv').query("status == 'in_workshop'").vehicle_id.iloc[0]
        changed.loc[changed.decision.eq('served').idxmax(), 'vehicle_id'] = workshop
        with self.assertRaises(ValueError):
            w.validate_allocation(self.data, changed)
        changed = sub.copy()
        changed.loc[0, 'outlet_id'] = 'WRONG'
        with self.assertRaises(ValueError):
            w.validate_allocation(self.data, changed)

    def test_saved_models_reproduce_files(self):
        out = Path(ARGS.output)
        if not all((out/'models'/name).exists() for name in ['task1_service.joblib','task1_lateness.joblib','task2a_total.joblib','task2a_chilled.joblib']):
            self.skipTest('Finish training all four saved models first')
        one = w.inference_task1(self.data, out/'models').set_index('delivery_id')
        expected = pd.read_csv(out/'submissions/submission_task1.csv').set_index('delivery_id')
        np.testing.assert_allclose(one.reindex(expected.index), expected, atol=1e-7, rtol=0)
        two = w.inference_task2a(self.data, out/'models').set_index('row_id')
        expected = pd.read_csv(out/'submissions/submission_task2a.csv').set_index('row_id')
        np.testing.assert_allclose(two.reindex(expected.index)[expected.columns], expected, atol=1e-7, rtol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='outputs')
    ARGS, remaining = parser.parse_known_args()
    unittest.main(argv=['test_solution.py'] + remaining)
