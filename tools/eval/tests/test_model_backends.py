import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import contextlib
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestModelBackends(unittest.TestCase):
    def test_detects_qwen_and_internvl_model_families(self):
        from models.utils.model_backends import detect_model_family

        self.assertEqual(
            detect_model_family("/data/ckpt/Qwen/Qwen2.5-VL-3B-Instruct"),
            "qwen2_5_vl",
        )
        self.assertEqual(
            detect_model_family("/data/ckpt/ReCogDrive-VLM-8B"),
            "internvl_chat",
        )
        self.assertEqual(
            detect_model_family("/data/ckpt/ReCogDrive-VLM-2B"),
            "internvl_chat",
        )

    def test_resolves_wrap_layer_classes_for_known_backends(self):
        from models.utils.model_backends import resolve_transformer_layer_classes

        qwen_classes = resolve_transformer_layer_classes("qwen2_5_vl")
        internvl_classes = resolve_transformer_layer_classes("internvl_chat")

        self.assertTrue(any(cls.__name__ == "Qwen2_5_VLDecoderLayer" for cls in qwen_classes))
        self.assertTrue(any(cls.__name__ == "Qwen2DecoderLayer" for cls in internvl_classes))

    def test_recogdrive_sft_config_uses_canonical_8_pose(self):
        import yaml

        config_path = REPO_ROOT / "config" / "training" / "recogdrive-vlm-8b-navsimv2-mix-sft-local8gpu.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["model"]["pretrained_model_path"], "/data/ckpt/ReCogDrive-VLM-8B")
        self.assertEqual(cfg["model"]["trajectory"]["num_poses"], 8)
        self.assertEqual(cfg["model"]["trajectory"]["interval_length"], 0.5)
        self.assertEqual(cfg["model"]["trajectory"]["time_horizon"], 4.0)
        self.assertTrue(cfg["model"]["lora"]["use"])
        self.assertTrue(cfg["training"]["gradient_checkpointing"])

    def test_recogdrive_2b_sft_config_uses_canonical_8_pose(self):
        import yaml

        config_path = REPO_ROOT / "config" / "training" / "recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["model"]["pretrained_model_path"], "/data/ckpt/ReCogDrive-VLM-2B")
        self.assertEqual(cfg["model"]["trajectory"]["num_poses"], 8)
        self.assertEqual(cfg["model"]["trajectory"]["interval_length"], 0.5)
        self.assertEqual(cfg["model"]["trajectory"]["time_horizon"], 4.0)
        self.assertFalse(cfg["model"]["lora"]["use"])
        self.assertFalse(cfg["training"]["gradient_checkpointing"])

    def test_resize_token_embeddings_falls_back_to_language_model(self):
        from models.utils.model_backends import resize_token_embeddings_for_model

        calls = []

        class LanguageModel:
            def resize_token_embeddings(self, size):
                calls.append(("language_model", size))
                return "ok"

        class TopLevelModel:
            def __init__(self):
                self.language_model = LanguageModel()

            def resize_token_embeddings(self, size):
                raise AttributeError("set_output_embeddings")

        result = resize_token_embeddings_for_model(TopLevelModel(), 123)

        self.assertEqual(result, "ok")
        self.assertEqual(calls, [("language_model", 123)])

    def test_load_causal_lm_for_model_uses_qwen_specific_class(self):
        from models.utils import model_backends

        with patch.object(model_backends, "detect_model_family", return_value="qwen2_5_vl"), patch.object(
            model_backends.Qwen2_5_VLForConditionalGeneration,
            "from_pretrained",
            return_value="qwen-model",
        ) as qwen_loader, patch.object(
            model_backends.AutoModelForCausalLM,
            "from_pretrained",
            return_value="auto-model",
        ) as auto_loader:
            model = model_backends.load_causal_lm_for_model("/fake/qwen", device="cpu")

        self.assertEqual(model, "qwen-model")
        qwen_loader.assert_called_once_with(
            "/fake/qwen",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        auto_loader.assert_not_called()

    def test_load_causal_lm_for_model_uses_auto_model_for_internvl(self):
        from models.utils import model_backends

        with patch.object(model_backends, "detect_model_family", return_value="internvl_chat"), patch.object(
            model_backends.Qwen2_5_VLForConditionalGeneration,
            "from_pretrained",
            return_value="qwen-model",
        ) as qwen_loader, patch.object(
            model_backends.AutoModelForCausalLM,
            "from_pretrained",
            return_value="internvl-model",
        ) as auto_loader:
            model = model_backends.load_causal_lm_for_model("/fake/internvl", device="cpu")

        self.assertEqual(model, "internvl-model")
        auto_loader.assert_called_once_with(
            "/fake/internvl",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        qwen_loader.assert_not_called()

    def test_resize_token_embeddings_prefers_top_level_when_supported(self):
        from models.utils.model_backends import resize_token_embeddings_for_model

        calls = []

        class LanguageModel:
            def resize_token_embeddings(self, size):
                calls.append(("language_model", size))
                return "language-ok"

        class TopLevelModel:
            def __init__(self):
                self.language_model = LanguageModel()

            def resize_token_embeddings(self, size):
                calls.append(("top_level", size))
                return "top-ok"

        result = resize_token_embeddings_for_model(TopLevelModel(), 456)

        self.assertEqual(result, "top-ok")
        self.assertEqual(calls, [("top_level", 456)])

    def test_gradient_checkpointing_prefers_top_level_when_supported(self):
        from models.utils.model_backends import set_gradient_checkpointing_for_model

        calls = []

        class LanguageModel:
            def gradient_checkpointing_enable(self):
                calls.append("language-enable")

            def gradient_checkpointing_disable(self):
                calls.append("language-disable")

        class TopLevelModel:
            def __init__(self):
                self.language_model = LanguageModel()

            def gradient_checkpointing_enable(self):
                calls.append("top-enable")

            def gradient_checkpointing_disable(self):
                calls.append("top-disable")

        model = TopLevelModel()
        set_gradient_checkpointing_for_model(model, enabled=True)
        set_gradient_checkpointing_for_model(model, enabled=False)

        self.assertEqual(calls, ["top-enable", "top-disable"])

    def test_gradient_checkpointing_falls_back_when_top_level_missing(self):
        from models.utils.model_backends import set_gradient_checkpointing_for_model

        calls = []

        class LanguageModel:
            def gradient_checkpointing_enable(self):
                calls.append("language-enable")

            def gradient_checkpointing_disable(self):
                calls.append("language-disable")

        class TopLevelModel:
            def __init__(self):
                self.language_model = LanguageModel()

        model = TopLevelModel()
        set_gradient_checkpointing_for_model(model, enabled=True)
        set_gradient_checkpointing_for_model(model, enabled=False)

        self.assertEqual(calls, ["language-enable", "language-disable"])

    def test_get_vision_backbone_prefers_top_level_vision_model(self):
        from models.utils.model_backends import get_vision_backbone

        vision = object()

        class TopLevelModel:
            def __init__(self):
                self.vision_model = vision
                self.base_model = SimpleNamespace(model=SimpleNamespace())

        resolved = get_vision_backbone(TopLevelModel())
        self.assertIs(resolved, vision)

    def test_get_vision_backbone_falls_back_to_nested_model(self):
        from models.utils.model_backends import get_vision_backbone

        vision = object()
        nested = SimpleNamespace(vision_model=vision)
        wrapper = SimpleNamespace(base_model=SimpleNamespace(model=nested))

        resolved = get_vision_backbone(wrapper)
        self.assertIs(resolved, vision)

    def test_align_vision_tensor_dtypes_casts_pixel_values_only(self):
        from models.utils.model_backends import align_vision_tensor_dtypes

        batch = {
            "pixel_values": torch.randn(2, 3, 4, 4, dtype=torch.float32),
            "pixel_values_videos": torch.randn(2, 4, 3, 4, 4, dtype=torch.float32),
            "input_ids": torch.ones(2, 5, dtype=torch.long),
        }

        aligned = align_vision_tensor_dtypes(batch, torch.bfloat16)

        self.assertEqual(aligned["pixel_values"].dtype, torch.bfloat16)
        self.assertEqual(aligned["pixel_values_videos"].dtype, torch.bfloat16)
        self.assertEqual(aligned["input_ids"].dtype, torch.long)

    def test_initialize_model_runtime_state_sets_internvl_img_context_token_id(self):
        from models.utils.model_backends import initialize_model_runtime_state

        processor = SimpleNamespace(
            img_context_token="<IMG_CONTEXT>",
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 123 if token == "<IMG_CONTEXT>" else -1),
        )
        model = SimpleNamespace(img_context_token_id=None)

        initialize_model_runtime_state(model, processor)

        self.assertEqual(model.img_context_token_id, 123)

    def test_maybe_wrap_with_lora_is_noop_when_disabled(self):
        from models.utils.model_backends import maybe_wrap_with_lora

        model = object()

        wrapped, enabled = maybe_wrap_with_lora(model, {"use": False})

        self.assertIs(wrapped, model)
        self.assertFalse(enabled)

    def test_maybe_wrap_with_lora_builds_expected_config(self):
        from models.utils import model_backends

        model = object()
        recorded = {}

        class FakeTaskType:
            CAUSAL_LM = "causal"

            def __getitem__(self, key):
                return getattr(self, key)

        def fake_lora_config(**kwargs):
            recorded["kwargs"] = kwargs
            return kwargs

        def fake_get_peft_model(vlm, lora_config):
            recorded["vlm"] = vlm
            recorded["lora_config"] = lora_config
            return "wrapped-model"

        with patch.object(model_backends, "TaskType", FakeTaskType()), patch.object(
            model_backends, "LoraConfig", side_effect=fake_lora_config
        ), patch.object(model_backends, "get_peft_model", side_effect=fake_get_peft_model):
            wrapped, enabled = model_backends.maybe_wrap_with_lora(
                model,
                {
                    "use": True,
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "bias": "none",
                },
            )

        self.assertTrue(enabled)
        self.assertEqual(wrapped, "wrapped-model")
        self.assertIs(recorded["vlm"], model)
        self.assertEqual(
            recorded["kwargs"],
            {
                "task_type": "causal",
                "target_modules": ["q_proj", "v_proj"],
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
            },
        )
        self.assertEqual(recorded["lora_config"], recorded["kwargs"])

    def test_maybe_wrap_with_lora_wraps_internvl_language_model(self):
        from models.utils import model_backends

        language_model = object()
        vlm = SimpleNamespace(config=SimpleNamespace(model_type="internvl_chat"), language_model=language_model)
        recorded = {}

        class FakeTaskType:
            CAUSAL_LM = "causal"

            def __getitem__(self, key):
                return getattr(self, key)

        def fake_lora_config(**kwargs):
            recorded["kwargs"] = kwargs
            return kwargs

        def fake_get_peft_model(model, lora_config):
            recorded["wrapped_model"] = model
            recorded["lora_config"] = lora_config
            return "wrapped-language-model"

        with patch.object(model_backends, "TaskType", FakeTaskType()), patch.object(
            model_backends, "LoraConfig", side_effect=fake_lora_config
        ), patch.object(model_backends, "get_peft_model", side_effect=fake_get_peft_model):
            wrapped_vlm, enabled = model_backends.maybe_wrap_with_lora(
                vlm,
                {
                    "use": True,
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj"],
                    "r": 8,
                    "alpha": 8,
                    "dropout": 0.1,
                    "bias": "none",
                },
            )

        self.assertTrue(enabled)
        self.assertIs(wrapped_vlm, vlm)
        self.assertEqual(wrapped_vlm.language_model, "wrapped-language-model")
        self.assertIs(recorded["wrapped_model"], language_model)

    def test_silence_internvl_runtime_prints_filters_dynamic_vit_messages(self):
        from models.utils.model_backends import silence_internvl_runtime_prints

        class DummyInternVL:
            def __init__(self):
                self.config = SimpleNamespace(model_type="internvl_chat")

        model = DummyInternVL()
        silence_internvl_runtime_prints(model)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print("dynamic ViT batch size: 12, images per sample: 12.0, dynamic token length: 3350")
            print("ordinary log line")

        self.assertEqual(buf.getvalue().strip(), "ordinary log line")

    def test_filter_generate_inputs_for_internvl_drops_image_flags(self):
        from models.utils.model_backends import filter_generate_inputs_for_model

        vlm = SimpleNamespace(config=SimpleNamespace(model_type="internvl_chat"))
        image_flags = torch.ones((2, 1), dtype=torch.long)
        pixel_values = torch.randn(2, 3, 4, 4)
        filtered = filter_generate_inputs_for_model(
            vlm,
            {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": pixel_values,
                "image_flags": image_flags,
            },
        )

        self.assertIn("input_ids", filtered)
        self.assertIn("pixel_values", filtered)
        self.assertNotIn("image_flags", filtered)

    def test_filter_generate_inputs_for_qwen_keeps_image_flags(self):
        from models.utils.model_backends import filter_generate_inputs_for_model

        vlm = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))
        image_flags = torch.ones((2, 1), dtype=torch.long)
        filtered = filter_generate_inputs_for_model(
            vlm,
            {
                "input_ids": torch.tensor([[1, 2]]),
                "image_flags": image_flags,
            },
        )

        self.assertIs(filtered["image_flags"], image_flags)

    def test_resolve_generate_length_kwargs_uses_max_new_tokens_for_internvl(self):
        from models.utils.model_backends import resolve_generate_length_kwargs

        vlm = SimpleNamespace(config=SimpleNamespace(model_type="internvl_chat"))

        resolved = resolve_generate_length_kwargs(
            vlm,
            configured_max_new_tokens=None,
            configured_max_length=2048,
        )

        self.assertEqual(resolved, {"max_new_tokens": 2048})

    def test_resolve_generate_length_kwargs_prefers_existing_max_new_tokens_for_internvl(self):
        from models.utils.model_backends import resolve_generate_length_kwargs

        vlm = SimpleNamespace(config=SimpleNamespace(model_type="internvl_chat"))

        resolved = resolve_generate_length_kwargs(
            vlm,
            configured_max_new_tokens=256,
            configured_max_length=2048,
        )

        self.assertEqual(resolved, {"max_new_tokens": 256})

    def test_resolve_generate_length_kwargs_keeps_max_length_for_qwen(self):
        from models.utils.model_backends import resolve_generate_length_kwargs

        vlm = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"))

        resolved = resolve_generate_length_kwargs(
            vlm,
            configured_max_new_tokens=None,
            configured_max_length=2048,
        )

        self.assertEqual(resolved, {"max_length": 2048})

    def test_get_input_ids_tensor_reads_mapping_style_batch(self):
        from models.utils.model_backends import get_input_ids_tensor

        input_ids = torch.tensor([[1, 2, 3]])

        resolved = get_input_ids_tensor({"input_ids": input_ids})

        self.assertIs(resolved, input_ids)

    def test_get_input_ids_tensor_reads_attr_style_batch(self):
        from models.utils.model_backends import get_input_ids_tensor

        input_ids = torch.tensor([[4, 5, 6]])

        resolved = get_input_ids_tensor(SimpleNamespace(input_ids=input_ids))

        self.assertIs(resolved, input_ids)


if __name__ == "__main__":
    unittest.main()
