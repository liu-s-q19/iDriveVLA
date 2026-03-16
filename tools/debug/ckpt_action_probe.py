#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from dataset_utils.sft_dataset import DataCollator, SFTDataset
from models.autovla import SFTAutoVLA


def parse_args():
    parser = argparse.ArgumentParser(description="Probe generated action tokens from a saved SFT checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml",
        help="Path to training config yaml.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which data split in config.data to probe.",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to probe.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation max_new_tokens. Defaults to training.generated_action_probe.max_new_tokens.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="How many generated text chars to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)

    config = yaml.safe_load(cfg_path.read_text())

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    using_cot = bool(config["model"]["use_cot"])
    split_cfg = config["data"][args.split]

    dataset = SFTDataset(split_cfg, config["model"], processor, using_cot=using_cot)
    collator = DataCollator(
        processor=processor,
        ignore_index=config["model"]["tokens"]["ignore_index"],
        assistant_id=config["model"]["tokens"]["assistant_id"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)

    model = SFTAutoVLA(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[info] dataset_split={args.split} dataset_size={len(dataset)}")
    print(f"[info] checkpoint={ckpt_path}")
    print(f"[info] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    device = torch.device(args.device)
    model.eval()
    model.autovla.vlm.to(device)
    model.autovla.device = str(device)
    model._assistant_id = model._assistant_id.to(device)

    probe_cfg = config.get("training", {}).get("generated_action_probe", {})
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(probe_cfg.get("max_new_tokens", 128))
    do_sample = bool(probe_cfg.get("do_sample", False))
    temperature = float(probe_cfg.get("temperature", 1.0))
    top_k = int(probe_cfg.get("top_k", 0))
    top_p = float(probe_cfg.get("top_p", 1.0))

    gen_kwargs = {"max_new_tokens": max(1, int(max_new_tokens)), "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_k": top_k, "top_p": top_p})
    print(f"[info] gen_kwargs={gen_kwargs}")

    end_index = args.start_index + max(0, args.num_samples)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx < args.start_index:
                continue
            if idx >= end_index:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_idx = model._find_subsequence_start(input_ids[0], model._assistant_id)
            if start_idx is None:
                print(f"[sample {idx}] assistant pattern not found")
                continue

            prompt_len = int(start_idx + model._assistant_id.numel())
            generate_inputs = {
                "input_ids": input_ids[:1, :prompt_len],
                "attention_mask": attention_mask[:1, :prompt_len],
            }

            batch_size = int(input_ids.shape[0])
            for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    if value.ndim > 0 and int(value.shape[0]) == batch_size:
                        generate_inputs[key] = value[:1]
                    else:
                        generate_inputs[key] = value

            outputs = model.autovla.vlm.generate(**generate_inputs, **gen_kwargs)
            completion_ids = outputs[:, prompt_len:][0]
            action_candidates = int((completion_ids >= model.autovla.action_start_id).sum().item())
            action_tokens = model._extract_action_tokens_from_completion(
                completion_ids, model.autovla.processor.tokenizer
            )

            completion_len = int(completion_ids.numel())
            action_len = int(action_tokens.numel())
            has_action = int(action_len > 0)

            text = model.autovla.processor.decode(completion_ids, skip_special_tokens=False)
            text_preview = text[: max(1, int(args.preview_chars))].replace("\n", "\\n")

            print(
                f"[sample {idx}] prompt_len={prompt_len} completion_len={completion_len} "
                f"action_candidates={action_candidates} action_len={action_len} has_action={has_action}"
            )
            print(f"[sample {idx}] first_action_ids={action_tokens[:10].tolist()}")
            print(f"[sample {idx}] preview={text_preview}")

    print("[done] probe finished")


if __name__ == "__main__":
    main()
