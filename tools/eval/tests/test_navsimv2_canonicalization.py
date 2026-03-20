import unittest
from unittest.mock import patch
import importlib.util
import sys
from pathlib import Path

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def _load_local_autovla_agent_module():
    module_path = Path(__file__).resolve().parents[3] / "navsim" / "navsim" / "agents" / "autovla_agent.py"
    spec = importlib.util.spec_from_file_location("test_local_autovla_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AUTOVLA_AGENT_MODULE = _load_local_autovla_agent_module()
AutoVLAAgentFeatureBuilder = _AUTOVLA_AGENT_MODULE.AutoVLAAgentFeatureBuilder
TrajectoryTargetBuilder = _AUTOVLA_AGENT_MODULE.TrajectoryTargetBuilder


class _FakeTokenProcessor:
    def __init__(self, *args, **kwargs):
        self.last_traj = None

    def __call__(self, traj):
        self.last_traj = traj.clone()
        return {
            "gt_idx": torch.zeros((1, traj.shape[0]), dtype=torch.long),
            "gt_pos_raw": traj[:, :2],
        }


class TestNavSimV2Canonicalization(unittest.TestCase):
    def test_feature_builder_truncates_gt_and_history_to_configured_num_poses(self):
        builder = AutoVLAAgentFeatureBuilder(sensor_data_path="/tmp/sensors", trajectory_num_poses=8)
        scene_data = {
            "instruction": "KEEP FORWARD",
            "velocity": [1.0, 0.0],
            "acceleration": [0.0, 0.0],
            "dataset_name": "navsim",
            "gt_trajectory": [[float(i), 0.0, 0.0] for i in range(10)],
            "front_camera_paths": ["f1", "f2", "f3", "f4"],
            "front_left_camera_paths": ["fl1", "fl2", "fl3", "fl4"],
            "front_right_camera_paths": ["fr1", "fr2", "fr3", "fr4"],
            "back_camera_paths": ["b1", "b2", "b3", "b4"],
            "back_left_camera_paths": ["bl1", "bl2", "bl3", "bl4"],
            "back_right_camera_paths": ["br1", "br2", "br3", "br4"],
        }

        features = builder.compute_features(scene_data)

        self.assertEqual(len(features["gt_trajectory"]), 8)
        self.assertEqual(len(features["history_trajectory"]), 8)

    def test_target_builder_tokenizes_only_first_num_poses(self):
        fake_token_processor = _FakeTokenProcessor()
        with patch.object(_AUTOVLA_AGENT_MODULE, "TokenProcessor", return_value=fake_token_processor):
            builder = TrajectoryTargetBuilder(
                trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5),
                codebook_cache_path="/tmp/fake.pkl",
            )

        builder.token_processor = fake_token_processor
        builder.compute_targets({"gt_trajectory": [[float(i), 0.0, 0.0] for i in range(10)]})

        self.assertIsNotNone(fake_token_processor.last_traj)
        self.assertEqual(fake_token_processor.last_traj.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
