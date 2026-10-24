import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import numpy as np
    import torch
    from app.rl.env import OBS_DIM, RacingVecEnv, ResetConfig
    from app.rl.ppo import ActorCritic, PPOConfig, RolloutBuffer, ppo_update
    from app.rl.policy_io import load_policy
    from app.track_geometry import load_track
    RL_AVAILABLE = True
except ModuleNotFoundError:
    RL_AVAILABLE = False


@unittest.skipUnless(RL_AVAILABLE, "install backend/requirements-rl.txt to run RL tests")
class RacingSelfPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = load_track("Silverstone")

    def make_env(self, **kwargs):
        return RacingVecEnv(
            [self.track],
            num_envs=1,
            cars_per_race=kwargs.pop("cars_per_race", 4),
            seed=kwargs.pop("seed", 17),
            **kwargs,
        )

    def test_seeded_grid_is_reproducible_and_cars_start_apart(self):
        config = ResetConfig(randomize_along=True, speed_fraction=(0.0, 0.0))
        first = self.make_env(reset_config=config)
        second = self.make_env(reset_config=config)
        first.reset()
        second.reset()
        np.testing.assert_allclose(first.x, second.x)
        np.testing.assert_allclose(first.y, second.y)
        distances = [
            math.hypot(first.x[0, a] - first.x[0, b], first.y[0, a] - first.y[0, b])
            for a in range(4) for b in range(a + 1, 4)
        ]
        self.assertGreater(min(distances), 4.5)
        self.assertEqual(len({first.track_idx[0], second.track_idx[0]}), 1)

    def test_observation_contains_opponents_and_masks_retired_car(self):
        env = self.make_env(cars_per_race=3)
        obs = env.reset()
        self.assertEqual(obs.shape, (3, OBS_DIM))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertGreaterEqual(obs[0, 25 + 5], 0.0)
        self.assertEqual(obs[0, 25 + 5], 1.0)
        env.active[0, 1] = False
        obs = env.observe()
        self.assertIn(0.0, obs[0, 25:OBS_DIM])

    def test_collision_terminates_both_cars_and_penalizes_them(self):
        env = self.make_env(cars_per_race=2, reset_config=ResetConfig(
            randomize_along=False, speed_fraction=(0.0, 0.0), heading_noise_rad=0.0,
        ), auto_reset=False)
        env.reset()
        env.x[0, 1] = env.x[0, 0]
        env.y[0, 1] = env.y[0, 0]
        env.heading[0, 1] = env.heading[0, 0]
        _, rewards, dones, infos = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual([infos[0]["reason"], infos[1]["reason"]], ["collision", "collision"])
        self.assertTrue(np.all(rewards < 0))
        self.assertEqual(env.collision_count[0], 1)

    def test_off_track_retirement_is_per_car(self):
        env = self.make_env(cars_per_race=2, reset_config=ResetConfig(
            randomize_along=False, speed_fraction=(0.0, 0.0), heading_noise_rad=0.0,
        ), auto_reset=False)
        env.reset()
        env.x[0, 0] = 10000.0
        env.y[0, 0] = 10000.0
        _, rewards, dones, infos = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(infos[0]["reason"], "off_track")
        self.assertLess(rewards[0], rewards[1])
        self.assertTrue(env.active[0, 1])

    def test_finish_and_timeout_are_reported_with_race_order(self):
        env = self.make_env(cars_per_race=2, reset_config=ResetConfig(
            randomize_along=False, speed_fraction=(0.0, 0.0), heading_noise_rad=0.0,
        ), auto_reset=False, max_steps=1)
        env.reset()
        ft = env.tracks[0]
        env.speed[0, 0] = 60.0
        env.lap_progress[0, 0] = ft.race_length_m * 0.999 - 0.01
        env.active[0, 1] = False
        env.reason[0, 1] = "collision"
        _, rewards, dones, infos = env.step(np.zeros((2, 2), dtype=np.float32))
        result = infos[0]["race_result"]
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual(result["finishes"], 1)
        self.assertEqual(result["cars"][0]["status"], "finished")
        self.assertEqual(result["cars"][1]["status"], "collision")
        self.assertGreater(rewards[0], 0.0)

        timeout = self.make_env(cars_per_race=2, reset_config=ResetConfig(
            randomize_along=False, speed_fraction=(0.0, 0.0), heading_noise_rad=0.0,
        ), auto_reset=False, max_steps=1)
        timeout.reset()
        _, _, _, timeout_infos = timeout.step(np.zeros((2, 2), dtype=np.float32))
        timeout_result = timeout_infos[0]["race_result"]
        self.assertEqual(timeout_result["finishes"], 0)
        self.assertTrue(all(car["status"] == "time" for car in timeout_result["cars"]))

    def test_short_ppo_update_filters_inactive_agents(self):
        policy = ActorCritic(obs_dim=OBS_DIM)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        buffer = RolloutBuffer(2, 4, OBS_DIM)
        obs = np.zeros((4, OBS_DIM), dtype=np.float32)
        actions = np.zeros((4, 2), dtype=np.float32)
        with torch.inference_mode():
            action_t, log_prob, value, _ = policy.act(torch.as_tensor(obs))
        actions = action_t.numpy()
        buffer.add(obs, actions, log_prob.numpy(), np.ones(4), np.zeros(4), value.numpy(),
                   valid=np.array([True, True, False, False]))
        buffer.add(obs, actions, log_prob.numpy(), np.ones(4), np.zeros(4), value.numpy(),
                   valid=np.array([True, True, False, False]))
        advantages, returns = buffer.advantages(np.zeros(4), np.zeros(4), 0.995, 0.95)
        stats = ppo_update(policy, optimizer, buffer, advantages, returns, PPOConfig(epochs=1, minibatches=2), torch.device("cpu"))
        self.assertGreater(stats["updates"], 0)
        self.assertTrue(all(math.isfinite(stats[key]) for key in ("policy_loss", "value_loss", "entropy")))

    def test_checkpoint_rejects_inconsistent_race_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-race.pt"
            policy = ActorCritic(obs_dim=OBS_DIM)
            torch.save({"obs_dim": OBS_DIM, "cars_per_race": 1, "policy": policy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Race checkpoints must use obs_dim=31"):
                load_policy(path)

    def test_checkpoint_versions_and_multi_car_telemetry(self):
        from app.simulation import SimulationEngine
        from app.rl.train import _checkpoint_payload

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_policy = ActorCritic(obs_dim=OBS_DIM)
            new_path = root / "self-play.pt"
            optimizer = torch.optim.Adam(new_policy.parameters())
            payload = _checkpoint_payload(
                new_policy, optimizer, Namespace(cars_per_race=4, num_races=8),
                ["Silverstone"], iteration=1, global_step=32,
            )
            self.assertEqual(payload["observation_layout"]["total_features"], OBS_DIM)
            self.assertEqual(payload["race_settings"]["laps"], 1)
            torch.save(payload, new_path)
            loaded = load_policy(new_path)
            self.assertEqual(loaded.obs_dim, OBS_DIM)
            self.assertEqual(loaded.cars_per_race, 4)
            engine = SimulationEngine(self.track)
            info = engine.load_checkpoint(new_path)
            self.assertEqual(info["race_mode"], "shared_policy_self_play")
            telemetry = engine.snapshot()
            self.assertEqual(len(telemetry["vehicles"]), 4)
            self.assertIn("vehicle", telemetry)
            self.assertIn("progress", telemetry)
            for _ in range(3):
                engine.step()
            self.assertEqual(len(engine.snapshot()["vehicles"]), 4)

            solo_policy = ActorCritic(obs_dim=25)
            solo_path = root / "legacy.pt"
            torch.save({"obs_dim": 25, "policy": solo_policy.state_dict()}, solo_path)
            legacy = load_policy(solo_path)
            self.assertEqual(legacy.obs_dim, 25)
            engine.load_checkpoint(solo_path)
            self.assertIsNone(engine.race_env)
            self.assertEqual(len(engine.snapshot()["vehicles"]), 1)


if __name__ == "__main__":
    unittest.main()
