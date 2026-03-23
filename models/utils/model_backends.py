from __future__ import annotations

import builtins
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - exercised through runtime environment
    LoraConfig = None
    TaskType = None
    get_peft_model = None


_INTERNVL_PRINT_FILTER_INSTALLED = False
_CHECKPOINT_WARNING_FILTER_INSTALLED = False
_INTERNVL_SUPPRESSED_PREFIXES = (
    "dynamic ViT batch size:",
)


def detect_model_family(model_path: str) -> str:
    path = str(Path(model_path))
    trust_remote_code = "ReCogDrive-VLM" in path or "InternVL" in path
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    return str(getattr(cfg, "model_type", "")).lower()


def resolve_transformer_layer_classes(model_family: str) -> Tuple[type, ...]:
    family = str(model_family).lower()
    if family == "qwen2_5_vl":
        return (Qwen2_5_VLDecoderLayer,)
    if family == "internvl_chat":
        return (Qwen2DecoderLayer,)
    raise ValueError(f"Unsupported model_family={model_family}")


def infer_internvl_num_image_token(model_path: str) -> int:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    image_size = cfg.force_image_size or cfg.vision_config.image_size
    patch_size = cfg.vision_config.patch_size
    return int((image_size // patch_size) ** 2 * (cfg.downsample_ratio ** 2))


def load_processor_for_model(model_path: str):
    family = detect_model_family(model_path)
    if family == "qwen2_5_vl":
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        processor.family = family
        return processor
    if family == "internvl_chat":
        return InternVLProcessorAdapter(model_path=model_path)
    raise ValueError(f"Unsupported model family for processor loading: {family}")


def load_causal_lm_for_model(model_path: str, device="cpu"):
    family = detect_model_family(model_path)
    if family == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    if family == "internvl_chat":
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    raise ValueError(f"Unsupported model family for causal LM loading: {family}")


def get_vision_backbone(vlm):
    if hasattr(vlm, "visual"):
        return vlm.visual
    if hasattr(vlm, "vision_model"):
        return vlm.vision_model
    if hasattr(vlm, "base_model") and hasattr(vlm.base_model, "model"):
        return get_vision_backbone(vlm.base_model.model)
    raise AttributeError("Unable to locate vision backbone on VLM")


def get_language_backbone(vlm):
    if hasattr(vlm, "base_model") and hasattr(vlm.base_model, "model"):
        return get_language_backbone(vlm.base_model.model)
    if hasattr(vlm, "model"):
        return vlm.model
    if hasattr(vlm, "language_model"):
        return vlm.language_model
    raise AttributeError("Unable to locate language backbone on VLM")


def resize_token_embeddings_for_model(vlm, vocab_size: int):
    try:
        return vlm.resize_token_embeddings(vocab_size)
    except AttributeError as exc:
        if "set_output_embeddings" not in str(exc):
            raise

    for attr_name in ("language_model", "base_model"):
        inner = getattr(vlm, attr_name, None)
        if inner is None or not hasattr(inner, "resize_token_embeddings"):
            continue
        return inner.resize_token_embeddings(vocab_size)

    raise AttributeError(
        f"Unable to resize token embeddings for model type {type(vlm).__name__}"
    )


def set_gradient_checkpointing_for_model(vlm, enabled: bool):
    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"

    if hasattr(vlm, method_name):
        return getattr(vlm, method_name)()

    for attr_name in ("language_model", "base_model", "model"):
        inner = getattr(vlm, attr_name, None)
        if inner is None or not hasattr(inner, method_name):
            continue
        return getattr(inner, method_name)()

    raise AttributeError(
        f"Unable to set gradient checkpointing for model type {type(vlm).__name__}"
    )


def align_vision_tensor_dtypes(batch: dict, target_dtype: torch.dtype) -> dict:
    aligned = dict(batch)
    for key in ("pixel_values", "pixel_values_videos"):
        value = aligned.get(key)
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != target_dtype:
            aligned[key] = value.to(dtype=target_dtype)
    return aligned


def filter_generate_inputs_for_model(vlm, model_inputs: dict) -> dict:
    filtered = dict(model_inputs)
    model_type = str(getattr(getattr(vlm, "config", None), "model_type", "")).lower()
    if model_type == "internvl_chat":
        # InternVL's custom `.generate()` forwards kwargs to the inner language model's `.generate()`,
        # which rejects vision-only fields like `image_flags`.
        filtered.pop("image_flags", None)
    return filtered


def extract_static_model_inputs(model_inputs: dict) -> dict:
    """
    Extract non-text (mostly vision) tensors that should be forwarded alongside
    `input_ids` / `attention_mask` for both `.generate()` and `.forward()`.

    This avoids hard-coding Qwen2.5-VL-only keys (pixel_values_videos/video_grid_thw)
    and supports InternVL/ReCogDrive (pixel_values/image_grid_thw) as well.
    """
    if not isinstance(model_inputs, dict):
        return {}

    static_keys = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_flags",
    )
    extracted = {}
    for key in static_keys:
        value = model_inputs.get(key)
        if value is None:
            continue
        extracted[key] = value
    return extracted


def filter_forward_inputs_for_model(vlm, model_inputs: dict) -> dict:
    """
    Filter inputs to match the target model's `.forward()` signature.

    This is necessary because some wrapped models (e.g., PEFT wrappers) accept
    arbitrary kwargs and forward them to inner models that do not.
    """
    import inspect

    if not isinstance(model_inputs, dict):
        return {}

    target = vlm
    # PEFT models typically expose `.base_model.model` as the real implementation.
    base_model = getattr(target, "base_model", None)
    if base_model is not None:
        target = getattr(base_model, "model", base_model)

    forward = getattr(target, "forward", None)
    if forward is None:
        return dict(model_inputs)

    try:
        sig = inspect.signature(forward)
    except (TypeError, ValueError):
        return dict(model_inputs)

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(model_inputs)

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in model_inputs.items() if k in allowed}


def resolve_generate_length_kwargs(vlm, configured_max_new_tokens=None, configured_max_length=None) -> dict:
    model_type = str(getattr(getattr(vlm, "config", None), "model_type", "")).lower()
    if configured_max_new_tokens is not None:
        return {"max_new_tokens": int(configured_max_new_tokens)}
    if configured_max_length is None:
        return {}
    if model_type == "internvl_chat":
        # InternVL/ReCogDrive multimodal prompts can exceed the legacy total-length cap.
        # Passing max_length then produces negative remaining length inside HF generate().
        return {"max_new_tokens": int(configured_max_length)}
    return {"max_length": int(configured_max_length)}


def get_input_ids_tensor(batch):
    if isinstance(batch, dict):
        return batch["input_ids"]
    return getattr(batch, "input_ids")


def extract_completion_ids_from_generate_output(generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> torch.Tensor:
    if generated_ids.ndim != 2 or prompt_input_ids.ndim != 2:
        raise ValueError("generated_ids and prompt_input_ids must be rank-2 tensors")

    if generated_ids.shape[0] != prompt_input_ids.shape[0]:
        raise ValueError("generated_ids and prompt_input_ids must have the same batch dimension")

    prompt_len = int(prompt_input_ids.shape[1])
    prompt_input_ids_aligned = prompt_input_ids.to(device=generated_ids.device)
    if generated_ids.shape[1] >= prompt_len and torch.equal(generated_ids[:, :prompt_len], prompt_input_ids_aligned):
        return generated_ids[:, prompt_len:]
    return generated_ids


def normalize_generate_output_sequences(
    generated_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if generated_ids.ndim != 2 or prompt_input_ids.ndim != 2:
        raise ValueError("generated_ids and prompt_input_ids must be rank-2 tensors")
    if generated_ids.shape[0] != prompt_input_ids.shape[0]:
        raise ValueError("generated_ids and prompt_input_ids must have the same batch dimension")

    completion_ids = extract_completion_ids_from_generate_output(generated_ids, prompt_input_ids)
    prompt_len = int(prompt_input_ids.shape[1])
    prompt_input_ids_aligned = prompt_input_ids.to(device=generated_ids.device)

    if generated_ids.shape[1] >= prompt_len and torch.equal(generated_ids[:, :prompt_len], prompt_input_ids_aligned):
        prompt_completion_ids = generated_ids
    else:
        prompt_completion_ids = torch.cat([prompt_input_ids_aligned, completion_ids], dim=1)

    return prompt_completion_ids, completion_ids


def silence_internvl_runtime_prints(vlm) -> None:
    global _INTERNVL_PRINT_FILTER_INSTALLED

    model_type = str(getattr(getattr(vlm, "config", None), "model_type", "")).lower()
    if model_type != "internvl_chat":
        return
    if _INTERNVL_PRINT_FILTER_INSTALLED:
        return

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args:
            first = str(args[0])
            if any(first.startswith(prefix) for prefix in _INTERNVL_SUPPRESSED_PREFIXES):
                return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print
    _INTERNVL_PRINT_FILTER_INSTALLED = True


def _install_runtime_warning_filters() -> None:
    global _CHECKPOINT_WARNING_FILTER_INSTALLED

    if _CHECKPOINT_WARNING_FILTER_INSTALLED:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"None of the inputs have requires_grad=True\. Gradients will be None",
        category=UserWarning,
    )
    _CHECKPOINT_WARNING_FILTER_INSTALLED = True


def _iter_generation_config_holders(vlm):
    candidates = [
        vlm,
        getattr(vlm, "language_model", None),
        getattr(vlm, "model", None),
        getattr(vlm, "base_model", None),
    ]
    base_model = getattr(vlm, "base_model", None)
    if base_model is not None:
        candidates.append(getattr(base_model, "model", None))

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        obj_id = id(candidate)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield candidate


def _set_default_pad_token_id(vlm, processor) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    resolved_pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
    if resolved_pad_token_id is None:
        return

    if getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token_id = resolved_pad_token_id
        except Exception:
            pass

    for holder in _iter_generation_config_holders(vlm):
        generation_config = getattr(holder, "generation_config", None)
        if generation_config is None:
            continue
        if getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = int(resolved_pad_token_id)


def initialize_model_runtime_state(vlm, processor) -> None:
    _install_runtime_warning_filters()
    _set_default_pad_token_id(vlm, processor)
    silence_internvl_runtime_prints(vlm)
    if hasattr(vlm, "img_context_token_id") and getattr(vlm, "img_context_token_id", None) is None:
        img_context_token = getattr(processor, "img_context_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if img_context_token is not None and tokenizer is not None:
            vlm.img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)


def maybe_wrap_with_lora(vlm, lora_conf):
    if not lora_conf or not bool(lora_conf.get("use", False)):
        return vlm, False

    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required when model.lora.use=true")

    lora_config = LoraConfig(
        task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
        target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        r=int(lora_conf.get("r", 8)),
        lora_alpha=int(lora_conf.get("alpha", lora_conf.get("lora_alpha", 8))),
        lora_dropout=float(lora_conf.get("dropout", lora_conf.get("lora_dropout", 0.1))),
        bias=lora_conf.get("bias", "none"),
    )
    model_type = str(getattr(getattr(vlm, "config", None), "model_type", "")).lower()
    if model_type == "internvl_chat" and hasattr(vlm, "language_model"):
        vlm.language_model = get_peft_model(vlm.language_model, lora_config)
        return vlm, True
    return get_peft_model(vlm, lora_config), True


def _flatten_message_content_for_internvl(content: Iterable[dict]) -> str:
    parts: List[str] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            parts.append(str(item.get("text", "")))
        elif item_type == "video":
            frames = item.get("video", []) or []
            parts.append("".join("<image>\n" for _ in frames))
        elif item_type == "image":
            parts.append("<image>\n")
        else:
            parts.append(str(item))
    return "".join(parts).strip()


class InternVLProcessorAdapter:
    family = "internvl_chat"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.num_image_token = infer_internvl_num_image_token(model_path)
        self.image_size = 448
        self.img_context_token = "<IMG_CONTEXT>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, add_vision_id=False):
        _ = add_vision_id
        flat_messages = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = _flatten_message_content_for_internvl(content)
            flat_messages.append({"role": message["role"], "content": content})
        return self.tokenizer.apply_chat_template(
            flat_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def _encode_text_with_images(self, text: str, image_count: int) -> str:
        image_tokens = self.img_start_token + self.img_context_token * self.num_image_token + self.img_end_token
        encoded = str(text)
        for _ in range(int(image_count)):
            encoded = encoded.replace("<image>", image_tokens, 1)
        return encoded

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = str(image_path)
        if path.startswith("file://"):
            path = path[len("file://") :]
        image = Image.open(path).convert("RGB")
        return self._transform(image)

    def build_batch(self, text: List[str], image_paths: List[List[str]], padding=True, return_tensors="pt"):
        encoded_texts = [
            self._encode_text_with_images(sample_text, len(sample_paths))
            for sample_text, sample_paths in zip(text, image_paths)
        ]
        tokenized = self.tokenizer(
            encoded_texts,
            padding=padding,
            return_tensors=return_tensors,
        )

        flat_images: List[torch.Tensor] = []
        for sample_paths in image_paths:
            for path in sample_paths:
                flat_images.append(self._load_image(path))
        if flat_images:
            pixel_values = torch.stack(flat_images, dim=0)
            image_flags = torch.ones((len(flat_images), 1), dtype=torch.long)
        else:
            pixel_values = torch.empty((0, 3, self.image_size, self.image_size), dtype=torch.float32)
            image_flags = torch.empty((0, 1), dtype=torch.long)

        batch = dict(tokenized)
        batch["pixel_values"] = pixel_values
        batch["image_flags"] = image_flags
        return batch

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        _ = images
        image_paths = videos if videos is not None else []
        return self.build_batch(text=text, image_paths=image_paths, padding=padding, return_tensors=return_tensors)
