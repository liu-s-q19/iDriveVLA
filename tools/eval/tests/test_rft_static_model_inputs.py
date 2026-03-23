import torch


def test_extract_static_model_inputs_internvl_like():
    from models.utils.model_backends import extract_static_model_inputs

    model_inputs = {
        "input_ids": torch.ones((1, 8), dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "image_grid_thw": torch.zeros((1, 3), dtype=torch.long),
    }
    extracted = extract_static_model_inputs(model_inputs)
    assert "pixel_values" in extracted
    assert "image_grid_thw" in extracted
    assert "pixel_values_videos" not in extracted
    assert "video_grid_thw" not in extracted
    assert "input_ids" not in extracted
    assert "attention_mask" not in extracted


def test_extract_static_model_inputs_qwen_like():
    from models.utils.model_backends import extract_static_model_inputs

    model_inputs = {
        "pixel_values_videos": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "video_grid_thw": torch.zeros((1, 3), dtype=torch.long),
        "image_flags": torch.zeros((1,), dtype=torch.long),
    }
    extracted = extract_static_model_inputs(model_inputs)
    assert "pixel_values_videos" in extracted
    assert "video_grid_thw" in extracted
    assert "image_flags" in extracted


def test_filter_forward_inputs_for_model_filters_unknown_kwargs():
    from models.utils.model_backends import filter_forward_inputs_for_model

    class Inner:
        def forward(self, input_ids, attention_mask, pixel_values=None):  # noqa: ANN001
            return None

    class Wrapper:
        def __init__(self):
            self.base_model = type("Base", (), {"model": Inner()})()

    vlm = Wrapper()
    inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "inputs_embeds": torch.zeros((1, 3, 8), dtype=torch.float32),
    }
    filtered = filter_forward_inputs_for_model(vlm, inputs)
    assert "input_ids" in filtered
    assert "attention_mask" in filtered
    assert "pixel_values" in filtered
    assert "inputs_embeds" not in filtered
