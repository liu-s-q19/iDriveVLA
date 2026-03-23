import unittest

import torch

from models.autovla import AutoVLA


class TestAutoVLAPromptPrefix(unittest.TestCase):
    @staticmethod
    def _dummy_input_features():
        return {
            "images": {
                "front_camera": ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"],
                "front_left_camera": ["fl1.jpg", "fl2.jpg", "fl3.jpg", "fl4.jpg"],
                "front_right_camera": ["fr1.jpg", "fr2.jpg", "fr3.jpg", "fr4.jpg"],
            },
            "sensor_data_path": "/data/dataset/navsim",
            "vehicle_velocity": [1.0, 2.0],
            "vehicle_acceleration": [0.1, 0.2],
            "driving_command": "Go Straight",
        }

    def test_get_prompt_appends_answer_prefix_message_when_enabled(self):
        model = object.__new__(AutoVLA)
        model.video_conf = {"min_pixels": 109760, "max_pixels": 109760}
        model.use_cot = False
        model.model_family = "internvl_chat"
        model._force_action_answer_prefix = True

        captured = {}

        def _capture(messages):
            captured["messages"] = messages
            return {"messages": messages}

        model._build_internvl_inputs = _capture

        model.get_prompt(self._dummy_input_features())

        messages = captured["messages"]
        self.assertEqual(messages[-1]["role"], "assistant")
        content = messages[-1]["content"]
        self.assertIsInstance(content, list)
        self.assertIn("<answer>", content[0]["text"])
        self.assertIn("The final output trajectory is:", content[0]["text"])

    def test_get_prompt_keeps_user_last_message_when_prefix_disabled(self):
        model = object.__new__(AutoVLA)
        model.video_conf = {"min_pixels": 109760, "max_pixels": 109760}
        model.use_cot = False
        model.model_family = "internvl_chat"
        model._force_action_answer_prefix = False

        captured = {}

        def _capture(messages):
            captured["messages"] = messages
            return {"messages": messages}

        model._build_internvl_inputs = _capture

        model.get_prompt(self._dummy_input_features())

        messages = captured["messages"]
        self.assertEqual(messages[-1]["role"], "user")

    def test_protocol_completion_ids_prepends_prefix_tokens_when_enabled(self):
        model = object.__new__(AutoVLA)
        model._force_action_answer_prefix = True
        model._action_answer_protocol_enabled = True
        model._answer_prefix_token_ids = [11, 12, 13]

        completion_ids = torch.tensor([101, 102], dtype=torch.long)
        augmented = model._augment_completion_for_protocol(completion_ids)

        self.assertTrue(torch.equal(augmented, torch.tensor([11, 12, 13, 101, 102], dtype=torch.long)))

    def test_protocol_completion_ids_keeps_original_when_prefix_disabled(self):
        model = object.__new__(AutoVLA)
        model._force_action_answer_prefix = False
        model._action_answer_protocol_enabled = True
        model._answer_prefix_token_ids = [11, 12, 13]

        completion_ids = torch.tensor([101, 102], dtype=torch.long)
        augmented = model._augment_completion_for_protocol(completion_ids)

        self.assertTrue(torch.equal(augmented, completion_ids))


if __name__ == "__main__":
    unittest.main()
