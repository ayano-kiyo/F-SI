import math
import unittest

from app.dynamics import ControlInput, KinematicBicycle, VehicleState
from app.simulation import SimulationEngine
from app.track_geometry import DEFAULT_TRACK


class TrackGeometryTests(unittest.TestCase):
    def test_silverstone_data_is_closed_and_metric(self):
        self.assertGreater(len(DEFAULT_TRACK.points), 1000)
        self.assertGreater(len(DEFAULT_TRACK.race_line), 1000)
        self.assertGreater(DEFAULT_TRACK.length_m, 5500)
        self.assertLess(DEFAULT_TRACK.length_m, 6200)
        self.assertTrue(all(point.width_left > 0 and point.width_right > 0 for point in DEFAULT_TRACK.points))

    def test_projection_reports_signed_track_edges(self):
        point = DEFAULT_TRACK.points[300]
        on_track = DEFAULT_TRACK.project(point.x, point.y)
        self.assertLess(on_track.distance_m, 0.1)
        dx, dy = DEFAULT_TRACK.tangent_at(300)
        outside_left = DEFAULT_TRACK.project(point.x - dy * 25, point.y + dx * 25)
        self.assertGreater(outside_left.signed_lateral_m, outside_left.width_left_m)


class DynamicsTests(unittest.TestCase):
    def test_fixed_step_is_deterministic_and_clamps_controls(self):
        model = KinematicBicycle()
        initial = VehicleState(0.0, 0.0, 0.0)
        control = ControlInput(steering=2.0, throttle=2.0, brake=-1.0)
        first = model.step(initial, control, 1 / 60)
        second = model.step(initial, control, 1 / 60)
        self.assertEqual(first, second)
        self.assertGreater(first.x, 0.0)
        self.assertGreater(first.y, 0.0)
        self.assertGreater(first.speed_mps, 0.0)

    def test_brake_does_not_reverse_vehicle(self):
        state = VehicleState(0, 0, 0, speed_mps=0.2)
        result = KinematicBicycle().step(state, ControlInput(0, 0, 1), 1 / 60)
        self.assertEqual(result.speed_mps, 0.0)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = SimulationEngine(DEFAULT_TRACK)

    def setUp(self):
        self.sim.reset()

    def test_baseline_drives_and_emits_planner_trajectory(self):
        initial_x, initial_y = self.sim.vehicle.x, self.sim.vehicle.y
        for _ in range(600):
            self.sim.step()
        self.assertGreater(self.sim.vehicle.speed_mps, 10)
        self.assertGreater(math.dist((initial_x, initial_y), (self.sim.vehicle.x, self.sim.vehicle.y)), 10)
        telemetry = self.sim.snapshot()
        self.assertEqual(telemetry["controller"]["name"], "baseline")
        self.assertEqual(telemetry["planner"]["name"], "JEV")
        self.assertGreater(len(telemetry["planned_trajectory"]), 10)
        self.assertTrue(all("x" in point and "y" in point for point in telemetry["planned_trajectory"]))

    def test_pause_speed_reset_and_missing_nn_are_explicit(self):
        self.sim.paused = True
        self.sim.step()
        self.assertEqual(self.sim.sim_time_s, 0.0)
        self.sim.set_speed(2.0)
        self.assertEqual(self.sim.simulation_speed, 2.0)
        with self.assertRaisesRegex(ValueError, "no checkpoint"):
            self.sim.set_controller("neural_network")
        self.sim.reset()
        self.assertEqual(self.sim.sim_time_s, 0.0)
        self.assertEqual(self.sim.simulation_speed, 1.0)


if __name__ == "__main__":
    unittest.main()

class MultiTrackTests(unittest.TestCase):
    def test_track_catalog_has_expected_circuits(self):
        from app.track_geometry import available_track_names, list_track_metadata
        names = available_track_names()
        self.assertIn("Silverstone", names)
        self.assertIn("Monza", names)
        self.assertGreaterEqual(len(names), 20)
        metadata = {entry["name"]: entry for entry in list_track_metadata()}
        self.assertIn("Spa", metadata)
        self.assertGreater(metadata["Spa"]["length_m"], 5000)

    def test_engine_hot_swaps_between_circuits(self):
        from app.track_geometry import load_track
        sim = SimulationEngine(load_track("Silverstone"))
        sim.set_track(load_track("Monza"))
        self.assertEqual(sim.track.name, "Monza")
        self.assertEqual(sim.sim_time_s, 0.0)
        for _ in range(200):
            sim.step()
        telemetry = sim.snapshot()
        self.assertEqual(telemetry["track"]["name"], "Monza")
        self.assertGreater(sim.vehicle.speed_mps, 10)

    def test_unknown_track_is_rejected(self):
        from app.track_geometry import load_track
        with self.assertRaises(ValueError):
            load_track("../secret")
        with self.assertRaises(ValueError):
            load_track("NotATrack")
