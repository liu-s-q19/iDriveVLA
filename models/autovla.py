import torch
import os
import torch.distributed as dist
import hashlib
import re
from tqdm import tqdm
from typing import Dict, Any
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.distributed.fsdp import StateDictType
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from models.action_tokenizer import ActionTokenizer
from models.utils.action_answer_protocol import parse_action_answer_completion
from models.utils.action_answer_protocol import summarize_group_outcomes
from models.utils.model_backends import detect_model_family
from models.utils.model_backends import get_language_backbone
from models.utils.model_backends import get_vision_backbone
from models.utils.model_backends import load_processor_for_model
from models.utils.model_backends import load_causal_lm_for_model
from models.utils.model_backends import resize_token_embeddings_for_model
from models.utils.model_backends import align_vision_tensor_dtypes
from models.utils.model_backends import initialize_model_runtime_state
from models.utils.model_backends import filter_generate_inputs_for_model
from models.utils.model_backends import resolve_generate_length_kwargs
from models.utils.model_backends import get_input_ids_tensor
from models.utils.grpo_metrics import masked_token_mean
from models.utils.grpo_log_keys import progress_bar_metric_names
from transformers.modeling_outputs import CausalLMOutputWithPast
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

try:
    from models.utils.score import PDM_Reward
    _PDM_REWARD_IMPORT_ERROR = None
except Exception as exc:
    # SFT inference/evaluation should not hard-fail on GRPO-only reward dependencies.
    PDM_Reward = None
    _PDM_REWARD_IMPORT_ERROR = exc


_PROTOCOL_INVALID_REASON_CODES = {
    "": 0,
    "missing_answer_block": 1,
    "multiple_answer_blocks": 2,
    "answer_block_not_at_tail": 3,
    "action_count_mismatch": 4,
    "not_run": 5,
}


class GRPOAutoVLA(pl.LightningModule):
    @staticmethod
    def _progress_bar_metric_names():
        return progress_bar_metric_names()

    def __init__(self, config: dict, inference=False):
        super().__init__()
        self.cfg = config
        self.use_cot = config['model']['use_cot']
        self.save_hyperparameters()
        if PDM_Reward is None:
            raise ImportError(
                "Failed to import GRPO reward dependencies from models.utils.score. "
                f"Original error: {_PDM_REWARD_IMPORT_ERROR}"
            )

        # Load trajectory sampling from config or use default
        traj_conf = config['model']['trajectory']
        self.trajectory_sampling = TrajectorySampling(
            num_poses=traj_conf['num_poses'],
            interval_length=traj_conf['interval_length']
        )
        
        # Load token configs
        token_conf = config['model']['tokens']
        self.action_start_id = token_conf['action_start_id']
        self.assistant_id = torch.tensor(token_conf['assistant_id'])

        # Training model (wrapped by Lightning FSDPStrategy)
        self.autovla = AutoVLA(config)

        self.autovla.train()
        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']
        self._log_grouped_rewards = bool(config.get('training', {}).get('log_grouped_rewards', False))
        self._empty_cache_each_step = bool(config.get('training', {}).get('empty_cache_each_step', False))
        self._sampling_seed_mode = str(config.get('training', {}).get('sampling_seed_mode', 'step_rank')).lower()
        self._sampling_seed_base = int(config.get('training', {}).get('sampling_seed_base', 1234))
        self._last_sampling_seed = None
        self._debug_compare_outputs = bool(config.get('training', {}).get('debug_compare_outputs', False))
        self._debug_compare_steps = int(config.get('training', {}).get('debug_compare_steps', 3))
        self._debug_text_preview_chars = int(config.get('training', {}).get('debug_text_preview_chars', 120))
        self._action_answer_protocol_enabled = bool(
            config.get('model', {}).get('action_answer_protocol', {}).get('enabled', False)
        )
        advantage_cfg = config.get('rl', {}).get('advantage', {})
        self._adv_group_std_eps = float(advantage_cfg.get('group_std_eps', 1e-6))
        self._adv_fallback_mode = str(advantage_cfg.get('fallback_mode', 'reward')).lower()

        # online reference model.
        if not inference:
            self.reference_model = AutoVLA(config, inference=True)
            state_dict = torch.load(config['model']['sft_model_path'])["state_dict"]
            state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in state_dict.items()}
            self.reference_model.load_state_dict(state_dict, strict=False)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()  
            print(f"Using online reference model from {config['model']['sft_model_path']}")

        # sample generation config
        sample_conf = config['training']['sample']
        self._sample_generation_temperature = {
            "temperature": sample_conf['temperature'],
            "top_k": sample_conf['top_k'],
            "top_p": sample_conf['top_p'],
        }
        if sample_conf.get('max_new_tokens') is not None:
            self._sample_generation_temperature["max_new_tokens"] = int(sample_conf['max_new_tokens'])
        else:
            self._sample_generation_temperature["max_length"] = sample_conf['max_length']

        # reward function
        reward_cfg = config.get('rl', {}).get('reward', {})
        self.train_critic = PDM_Reward(Path(config['data']['train']['metric_cache_path']), reward_cfg=reward_cfg)
        self.val_critic = PDM_Reward(Path(config['data']['val']['metric_cache_path']), reward_cfg=reward_cfg)

        # sliding window for training reward
        if not inference:
            self.window_size = config['rl']['reward'].get("sliding_window_size", 100)
            self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
            self.register_buffer("sliding_idx",   torch.zeros(1, dtype=torch.long))
            self.register_buffer("window_count",  torch.zeros(1, dtype=torch.long))

    def training_step(self, batch):
        # Generate a sample from the model.
        self.autovla.train()
        with torch.no_grad():
            sample = self.generate_sample(
                batch, model=self.autovla, device=next(self.parameters()).device)
        
            # Compute the reward for the generated sample.
            reward = self.reward_function(sample)
            reward_scale = self.cfg['rl']['reward'].get("scale", 1.0)
            reward = reward * reward_scale
            self.log("scaled_train_reward", reward.mean(), sync_dist=True, prog_bar=("scaled_train_reward" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
            self._log_sample_health(sample, device=reward.device)
            
            # Normalize the rewards to compute the advantage.
            groupped_rewards = self.all_gather(reward).flatten()
            group_mean = groupped_rewards.mean()
            group_std = groupped_rewards.std(unbiased=False)
            local_valid = torch.tensor(
                1.0 if bool(sample.get("reward_input_valid", True)) else 0.0,
                device=reward.device,
            )
            grouped_valid = self.all_gather(local_valid).flatten()
            group_summary = summarize_group_outcomes(
                grouped_rewards=groupped_rewards,
                grouped_valid_mask=grouped_valid,
                group_std_eps=self._adv_group_std_eps,
            )
            grouped_seeds = None
            if self._log_grouped_rewards and self._last_sampling_seed is not None:
                local_seed = torch.tensor(float(self._last_sampling_seed), device=reward.device)
                grouped_seeds = self.all_gather(local_seed).flatten()
            if self._log_grouped_rewards and self.global_rank == 0:
                print(groupped_rewards)
                if grouped_seeds is not None:
                    print(f"group_sampling_seeds={grouped_seeds.tolist()}")
                print(f"group_reward_std={group_std.item():.6f}")
            self._debug_compare_group_outputs(sample, groupped_rewards, group_std)
            self.log("group_reward_std", group_std, sync_dist=False, prog_bar=("group_reward_std" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
            self.log("group_valid_count", group_summary["group_valid_count"], sync_dist=False, on_step=True, on_epoch=False)
            self.log("group_all_invalid", group_summary["group_all_invalid"], sync_dist=False, on_step=True, on_epoch=False)
            self.log("group_all_same_reward", group_summary["group_all_same_reward"], sync_dist=False, on_step=True, on_epoch=False)
            fallback_used = 0.0
            if group_summary["use_zero_advantage"]:
                advantage = torch.zeros_like(reward)
                fallback_used = 1.0
                if self._log_grouped_rewards and self.global_rank == 0:
                    print("group_advantage_fallback=zero")
            else:
                advantage = (reward - group_mean) / (group_std + 1e-4)
            self.log("group_adv_fallback", fallback_used, sync_dist=False, prog_bar=("group_adv_fallback" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
            self.log("train_advantage", advantage.mean(), sync_dist=True, on_step=True, on_epoch=False)

        # Compute the per-token log probabilities.
        per_token_logps = self.get_per_token_logps(
            self.autovla.vlm, 
            sample['input_ids'], 
            sample['attention_mask'], 
            sample['pixel_values_videos'], 
            sample['video_grid_thw']
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, sample['prompt_length']-1:]
        completion_mask = sample['completion_mask']

        # reference model
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(
                self.reference_model.vlm, 
                sample["input_ids"], 
                sample["attention_mask"], 
                sample["pixel_values_videos"], 
                sample["video_grid_thw"]
            )
            ref_per_token_logps = ref_per_token_logps[:, sample["prompt_length"]-1:]

        # Compute the policy loss
        per_policy_loss = \
            torch.exp(per_token_logps - per_token_logps.detach()) * advantage.unsqueeze(-1)
        policy_objective = masked_token_mean(per_policy_loss, completion_mask)
        policy_loss = -policy_objective

        # Compute the kl loss
        kl_beta = self.cfg['rl'].get("kl_beta", 0.0)
        per_token_kl = \
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_kl_loss = kl_beta * per_token_kl
        kl_loss = masked_token_mean(per_kl_loss, completion_mask)

        per_token_loss = -(per_policy_loss - per_kl_loss)
        loss = masked_token_mean(per_token_loss, completion_mask)

        # Log metrics
        self.log("loss", loss, sync_dist=True, prog_bar=("loss" in self._progress_bar_metric_names()))
        self.log("policy_objective", policy_objective, sync_dist=True, prog_bar=False, on_step=True, on_epoch=False)
        self.log("policy_loss", policy_loss, sync_dist=True, prog_bar=("policy_loss" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
        self.log("kl_divergence", kl_loss, sync_dist=True, on_step=True, on_epoch=False)

        # record training reward
        self.training_buffer_record(reward.mean())
        return loss
    
    def training_buffer_record(self, step_reward):
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "avg_train_reward",
                sliding_avg,
                sync_dist=False, 
                prog_bar=True
            )

    def _fallback_advantage(self, reward: torch.Tensor) -> torch.Tensor:
        if self._adv_fallback_mode == "zero":
            return torch.zeros_like(reward)
        if self._adv_fallback_mode == "running_baseline":
            baseline = self._running_reward_baseline(reward.device, reward.dtype)
            return reward - baseline
        # Default fallback: keep a non-zero learning signal when group std collapses.
        return reward

    def _running_reward_baseline(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(self.window_count.item())
        if count <= 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return self.training_reward_buffer[:count].to(device=device, dtype=dtype).mean()

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True)

    @staticmethod
    def _hash_bytes(raw: bytes) -> str:
        return hashlib.sha1(raw).hexdigest()[:16]

    def _hash_text(self, text: str) -> str:
        return self._hash_bytes(text.encode("utf-8", errors="ignore"))

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        arr = tensor.detach().cpu().contiguous().numpy()
        return self._hash_bytes(arr.tobytes())

    def _parse_action_answer_completion(self, completion_ids: torch.Tensor, tokenizer):
        return parse_action_answer_completion(
            completion_ids.detach().cpu(),
            tokenizer=tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=int(self.trajectory_sampling.num_poses),
        )

    @staticmethod
    def _invalid_reason_code(reason: str) -> int:
        return int(_PROTOCOL_INVALID_REASON_CODES.get(str(reason), max(_PROTOCOL_INVALID_REASON_CODES.values()) + 1))

    def _action_tokens_tensor(self, action_token_ids: List[int], device: torch.device) -> torch.Tensor:
        if not action_token_ids:
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.tensor(action_token_ids, dtype=torch.long, device=device)

    def _normalize_action_tokens_default(self, action_tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
        if len(action_tokens) > self.trajectory_sampling.num_poses:
            return action_tokens[: self.trajectory_sampling.num_poses]
        if len(action_tokens) < self.trajectory_sampling.num_poses:
            return torch.cat(
                [action_tokens, torch.zeros(self.trajectory_sampling.num_poses - len(action_tokens), device=device)]
            ).long()
        return action_tokens.long()

    def _zero_padded_action_tokens(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.trajectory_sampling.num_poses, dtype=torch.long, device=device)

    def _trajectory_from_action_tokens(self, action_tokens: torch.Tensor) -> Trajectory:
        decode_tokens = action_tokens.detach().cpu()
        decoded_traj = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(decode_tokens)[0, 1:]
        return Trajectory(decoded_traj.cpu().numpy(), self.trajectory_sampling)

    def _extract_action_tokens_from_completion(self, completion_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        valid_action_ids = []
        for tok in completion_ids.tolist():
            if tok < self.action_start_id:
                continue
            decoded = tokenizer.decode([int(tok)])
            m = re.fullmatch(r"<action_(\d+)>", decoded.strip())
            if m is not None:
                valid_action_ids.append(int(tok))
        if not valid_action_ids:
            return torch.empty((0,), dtype=torch.long, device=completion_ids.device)
        return torch.tensor(valid_action_ids, dtype=torch.long, device=completion_ids.device)

    def _should_debug_compare(self) -> bool:
        if not self._debug_compare_outputs:
            return False
        step = int(getattr(self, "global_step", 0))
        return step < self._debug_compare_steps

    def _debug_compare_group_outputs(self, sample: Dict[str, Any], grouped_rewards: torch.Tensor, group_std: torch.Tensor) -> None:
        if not self._should_debug_compare():
            return

        step = int(getattr(self, "global_step", 0))
        rank = int(getattr(self, "global_rank", 0))
        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
        else:
            world_size = int(getattr(self, "world_size", 1))

        local_summary = {
            "rank": rank,
            "step": step,
            "token": str(sample["token"][0]) if isinstance(sample.get("token"), list) and sample["token"] else str(sample.get("token")),
            "completion_text_hash": sample.get("completion_text_hash"),
            "completion_ids_hash": sample.get("completion_ids_hash"),
            "action_tokens_hash": sample.get("action_tokens_hash"),
            "trajectory_hash": sample.get("trajectory_hash"),
            "action_candidate_count": sample.get("action_candidate_count"),
            "action_tokens_len": sample.get("action_tokens_len"),
            "answer_action_tokens_len": sample.get("answer_action_tokens_len"),
            "answer_block_valid": sample.get("answer_block_valid"),
            "reward_invalid_reason": sample.get("reward_invalid_reason"),
            "action_nonzero_count": sample.get("action_nonzero_count"),
            "completion_preview": sample.get("completion_preview"),
            "sampling_seed": sample.get("sampling_seed"),
        }

        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_summary)
        else:
            gathered = [local_summary]

        if rank != 0:
            return

        print(f"[debug_compare] step={step} world_size={world_size}")
        print(f"[debug_compare] grouped_rewards={grouped_rewards.flatten().tolist()} group_std={float(group_std.item()):.6f}")
        for item in sorted(gathered, key=lambda x: x["rank"]):
            preview = item["completion_preview"] if item["completion_preview"] is not None else ""
            print(
                f"[debug_compare][rank={item['rank']}] seed={item['sampling_seed']} token={item['token']} "
                f"text_hash={item['completion_text_hash']} ids_hash={item['completion_ids_hash']} "
                f"action_hash={item['action_tokens_hash']} traj_hash={item['trajectory_hash']} "
                f"action_candidates={item['action_candidate_count']} action_len={item['action_tokens_len']} "
                f"answer_action_len={item['answer_action_tokens_len']} "
                f"answer_valid={item['answer_block_valid']} invalid_reason={item['reward_invalid_reason']} "
                f"action_nonzero={item['action_nonzero_count']} "
                f"preview={preview!r}"
            )

    def _log_sample_health(self, sample: Dict[str, Any], device: torch.device) -> None:
        action_candidates = float(sample.get("action_candidate_count", 0))
        action_len = float(sample.get("action_tokens_len", 0))
        action_nonzero = float(sample.get("action_nonzero_count", 0))
        has_action = 1.0 if action_len > 0 else 0.0
        has_answer_block = 1.0 if float(sample.get("answer_block_count", 0)) > 0 else 0.0
        multiple_answer_blocks = 1.0 if float(sample.get("answer_block_count", 0)) > 1 else 0.0
        answer_block_at_tail = float(sample.get("answer_block_at_tail", 0.0))
        answer_action_len = float(sample.get("answer_action_tokens_len", 0))
        answer_block_valid = float(sample.get("answer_block_valid", 0.0))
        reward_input_valid = float(sample.get("reward_input_valid", 1.0))
        reward_invalid_reason = float(self._invalid_reason_code(sample.get("reward_invalid_reason", "")))
        completion_ids = sample.get("completion_ids")
        if isinstance(completion_ids, torch.Tensor) and completion_ids.ndim >= 2:
            completion_len = float(completion_ids.shape[1])
        else:
            completion_len = 0.0
        prompt_len = float(sample.get("prompt_length", 0))

        self.log("sample_action_candidate_count", torch.tensor(action_candidates, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_action_tokens_len", torch.tensor(action_len, device=device), sync_dist=True, prog_bar=("sample_action_tokens_len" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
        self.log("sample_action_nonzero_count", torch.tensor(action_nonzero, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_has_action", torch.tensor(has_action, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_has_answer_block", torch.tensor(has_answer_block, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_multiple_answer_blocks", torch.tensor(multiple_answer_blocks, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_answer_block_at_tail", torch.tensor(answer_block_at_tail, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_answer_action_tokens_len", torch.tensor(answer_action_len, device=device), sync_dist=True, prog_bar=("sample_answer_action_tokens_len" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
        self.log("sample_answer_block_valid", torch.tensor(answer_block_valid, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("reward_input_valid", torch.tensor(reward_input_valid, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("reward_invalid_reason", torch.tensor(reward_invalid_reason, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_completion_len", torch.tensor(completion_len, device=device), sync_dist=True, on_step=True, on_epoch=False)
        self.log("sample_prompt_len", torch.tensor(prompt_len, device=device), sync_dist=True, on_step=True, on_epoch=False)
    
    def reward_function(self, sample):
        device = next(self.parameters()).device

        if not bool(sample.get("reward_input_valid", True)):
            reward = torch.tensor(0.0, device=device)
            cot_penalties = torch.tensor(0.0, device=device, dtype=reward.dtype)
            self.log("train_reward", reward, sync_dist=True, prog_bar=("train_reward" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
            self.log("cot_penalty", cot_penalties, sync_dist=True, prog_bar=("cot_penalty" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
            return reward

        # Add pdm score for nuplan scenario
        reward = self.train_critic.rl_pdm_score(sample['trajectory'], sample['token'])
        reward = torch.tensor(reward).to(device)

        # Add chain-of-thought penalty (if "need cot" is found in the generated text).
        if self.use_cot:
            cot_conf = self.cfg['rl']['cot_penalty']
            cot_penalty_coef = cot_conf['coef']
            center = cot_conf['center']
            cot_penalty_weight = cot_conf['weight']

            cot_penalties = torch.stack([
                torch.sigmoid(torch.tensor(
                    (len(text) - center) * cot_penalty_coef,
                    device=device,
                    dtype=reward.dtype
                ))
                if "complex scenario" in text.lower() else torch.tensor(
                    0.0, device=device, dtype=reward.dtype
                )
                for text in sample['completion_texts']
            ])
            reward = reward - cot_penalty_weight * cot_penalties
        else:
            cot_penalties = torch.tensor(0.0, device=device, dtype=reward.dtype)

        self.log("train_reward", reward, sync_dist=True, prog_bar=("train_reward" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)
        self.log("cot_penalty", cot_penalties.mean(), sync_dist=True, prog_bar=("cot_penalty" in self._progress_bar_metric_names()), on_step=True, on_epoch=False)

        return reward
    
    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        # Get the per-token log probabilities for the completions for the model and the reference model
        logits = model(input_ids, attention_mask=attention_mask, 
                       pixel_values_videos=pixel_values_videos, 
                       video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, L-1, V)
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        return per_token_logps

    def _seed_sampling_rng(self):
        if self._sampling_seed_mode == "none":
            self._last_sampling_seed = None
            return None

        trainer = getattr(self, "_trainer", None)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
        else:
            world_size = max(int(getattr(trainer, "world_size", 1)), 1) if trainer is not None else 1
            global_rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
        global_step = int(getattr(trainer, "global_step", 0)) if trainer is not None else 0

        if self._sampling_seed_mode == "rank":
            seed = self._sampling_seed_base + global_rank
        else:
            # Default: make sampling differ across both rank and step while staying reproducible.
            seed = self._sampling_seed_base + global_step * world_size + global_rank

        seed = int(seed % (2**31 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._last_sampling_seed = seed
        return seed

    def generate_sample(self, data, model, device):

        # Get the model inputs
        inputs = model.get_prompt(data['input_features'])
        model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        model_inputs = model._align_model_inputs(model_inputs)

        # Generate completion
        with torch.no_grad():
            gen_kwargs = {
                "do_sample": True,
                "temperature": self._sample_generation_temperature['temperature'],
                "top_k": self._sample_generation_temperature['top_k'],
                "top_p": self._sample_generation_temperature['top_p'],
            }
            self._seed_sampling_rng()
            gen_kwargs.update(
                resolve_generate_length_kwargs(
                    model.vlm,
                    configured_max_new_tokens=self._sample_generation_temperature.get('max_new_tokens'),
                    configured_max_length=self._sample_generation_temperature.get('max_length'),
                )
            )

            generate_inputs = filter_generate_inputs_for_model(model.vlm, model_inputs)
            prompt_completion_ids = model.vlm.generate(
                **generate_inputs,
                **gen_kwargs,
            )

            prompt_length = get_input_ids_tensor(inputs).size(1)
            prompt_mask = model_inputs['attention_mask']
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Extract action tokens and trajectory (! batch size = 1)
            raw_action_candidates = completion_ids[0][completion_ids[0] >= self.action_start_id]
            raw_action_candidate_count = int(raw_action_candidates.numel())
            global_action_tokens = self._extract_action_tokens_from_completion(completion_ids[0], model.processor.tokenizer)
            raw_action_len = int(global_action_tokens.numel())

            if self._action_answer_protocol_enabled:
                parse_result = self._parse_action_answer_completion(completion_ids[0], model.processor.tokenizer)
                if parse_result.is_valid:
                    actions_tokens = self._action_tokens_tensor(parse_result.action_token_ids, device=device)
                else:
                    actions_tokens = self._zero_padded_action_tokens(device=device)
                answer_block_count = int(parse_result.answer_block_count)
                answer_block_at_tail = 1.0 if parse_result.answer_block_at_tail else 0.0
                answer_action_tokens_len = int(parse_result.action_token_count)
                answer_block_valid = 1.0 if parse_result.is_valid else 0.0
                reward_input_valid = 1.0 if parse_result.is_valid else 0.0
                reward_invalid_reason = parse_result.invalid_reason
            else:
                actions_tokens = self._normalize_action_tokens_default(global_action_tokens, device=device)
                answer_block_count = 0
                answer_block_at_tail = 0.0
                answer_action_tokens_len = 0
                answer_block_valid = 0.0
                reward_input_valid = 1.0
                reward_invalid_reason = ""

            trajectory = self._trajectory_from_action_tokens(actions_tokens)
            decoded_traj_np = trajectory.poses

            # Create completion mask
            is_eos = completion_ids == model.processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) 

            completion_texts = model.processor.batch_decode(completion_ids)
            completion_text = completion_texts[0] if completion_texts else ""
            action_nonzero_count = int((actions_tokens != 0).sum().item())
            completion_preview = completion_text[:self._debug_text_preview_chars]
            completion_text_hash = self._hash_text(completion_text)
            completion_ids_hash = self._hash_tensor(completion_ids[0])
            action_tokens_hash = self._hash_tensor(actions_tokens)
            trajectory_hash = self._hash_bytes(decoded_traj_np.tobytes())

            # Create outputs
            outputs = {'trajectory': trajectory, 
                       'token': data['token'], 
                       'completion_texts': completion_texts,
                       'prompt_length': prompt_length,
                       'input_ids': prompt_completion_ids, 
                       'completion_ids': completion_ids,
                       'attention_mask': attention_mask,
                       'completion_mask': completion_mask,
                       'pixel_values_videos': model_inputs['pixel_values_videos'], 
                       'video_grid_thw': model_inputs['video_grid_thw'],
                       'completion_preview': completion_preview,
                       'completion_text_hash': completion_text_hash,
                       'completion_ids_hash': completion_ids_hash,
                       'action_tokens_hash': action_tokens_hash,
                       'action_candidate_count': raw_action_candidate_count,
                       'action_tokens_len': raw_action_len,
                       'answer_block_count': answer_block_count,
                       'answer_block_at_tail': answer_block_at_tail,
                       'answer_action_tokens_len': answer_action_tokens_len,
                       'answer_block_valid': answer_block_valid,
                       'reward_input_valid': reward_input_valid,
                       'reward_invalid_reason': reward_invalid_reason,
                       'action_nonzero_count': action_nonzero_count,
                       'trajectory_hash': trajectory_hash,
                       'sampling_seed': self._last_sampling_seed,
                        }
        
        if self._empty_cache_each_step:
            torch.cuda.empty_cache()

        return outputs
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in get_vision_backbone(self.autovla.vlm).parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in get_language_backbone(self.autovla.vlm).parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        lr = float(self.cfg['training']['learning_rate'])
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=lr,
            weight_decay=wd
        )

        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Filter out parameters with no gradient to avoid empty tensor lists
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=gradient_clip_val)

    def on_save_checkpoint(self, checkpoint: dict):
        # only save main model
        sd = checkpoint.get("state_dict", {})
        for k in list(sd):
            if k.startswith("reference_model."):
                sd.pop(k)

class SFTAutoVLA(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.autovla = AutoVLA(config)
        self.autovla.train()

        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']
        probe_cfg = config.get('training', {}).get('generated_action_probe', {})
        self._probe_generated_action_enabled = bool(probe_cfg.get('enabled', False))
        self._probe_generated_action_every_n_steps = int(probe_cfg.get('every_n_steps', 0))
        self._probe_generated_action_max_new_tokens = int(probe_cfg.get('max_new_tokens', 128))
        self._probe_generated_action_do_sample = bool(probe_cfg.get('do_sample', False))
        self._probe_generated_action_temperature = float(probe_cfg.get('temperature', 1.0))
        self._probe_generated_action_top_k = int(probe_cfg.get('top_k', 0))
        self._probe_generated_action_top_p = float(probe_cfg.get('top_p', 1.0))
        self._probe_generated_action_last_step = -1
        self._assistant_id = torch.tensor(config['model']['tokens']['assistant_id'], dtype=torch.long)

    @staticmethod
    def _find_subsequence_start(sequence: torch.Tensor, pattern: torch.Tensor):
        if sequence.ndim != 1 or pattern.ndim != 1:
            return None
        if pattern.numel() == 0 or sequence.numel() < pattern.numel():
            return None
        for idx in range(sequence.numel() - pattern.numel() + 1):
            if torch.equal(sequence[idx:idx + pattern.numel()], pattern):
                return idx
        return None

    def _extract_action_tokens_from_completion(self, completion_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        valid_action_ids = []
        for tok in completion_ids.tolist():
            if tok < self.autovla.action_start_id:
                continue
            decoded = tokenizer.decode([int(tok)])
            m = re.fullmatch(r"<action_(\d+)>", decoded.strip())
            if m is not None:
                valid_action_ids.append(int(tok))
        if not valid_action_ids:
            return torch.empty((0,), dtype=torch.long, device=completion_ids.device)
        return torch.tensor(valid_action_ids, dtype=torch.long, device=completion_ids.device)

    def _maybe_log_generated_action_probe(self, batch: Dict[str, torch.Tensor]) -> None:
        if not self._probe_generated_action_enabled:
            return
        if self._probe_generated_action_every_n_steps <= 0:
            return
        if int(getattr(self, "global_rank", 0)) != 0:
            return

        step = int(getattr(self, "global_step", 0))
        if step % self._probe_generated_action_every_n_steps != 0:
            return
        if step == self._probe_generated_action_last_step:
            return

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            return

        assistant_pattern = self._assistant_id.to(device=input_ids.device)
        start_idx = self._find_subsequence_start(input_ids[0], assistant_pattern)
        if start_idx is None:
            return
        prompt_len = int(start_idx + assistant_pattern.numel())

        generate_inputs = {
            "input_ids": input_ids[:1, :prompt_len],
            "attention_mask": attention_mask[:1, :prompt_len],
        }
        batch_size = int(input_ids.shape[0])
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            val = batch.get(key)
            if isinstance(val, torch.Tensor):
                # Qwen2.5-VL collator may flatten vision tensors without a leading batch dim.
                # Slice only tensors whose leading dim is the actual batch size.
                if val.ndim > 0 and int(val.shape[0]) == batch_size:
                    generate_inputs[key] = val[:1]
                else:
                    generate_inputs[key] = val

        gen_kwargs = {
            "max_new_tokens": max(1, int(self._probe_generated_action_max_new_tokens)),
            "do_sample": bool(self._probe_generated_action_do_sample),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs.update(
                {
                    "temperature": float(self._probe_generated_action_temperature),
                    "top_k": int(self._probe_generated_action_top_k),
                    "top_p": float(self._probe_generated_action_top_p),
                }
            )

        vlm = self.autovla.vlm
        was_training = vlm.training
        self._probe_generated_action_last_step = step
        try:
            vlm.eval()
            with torch.no_grad():
                generate_inputs = self.autovla._align_model_inputs(generate_inputs)
                generate_inputs = filter_generate_inputs_for_model(vlm, generate_inputs)
                prompt_completion_ids = vlm.generate(**generate_inputs, **gen_kwargs)
            completion_ids = prompt_completion_ids[:, prompt_len:][0]
            action_candidates = int((completion_ids >= self.autovla.action_start_id).sum().item())
            action_tokens = self._extract_action_tokens_from_completion(
                completion_ids, self.autovla.processor.tokenizer
            )
            action_len = int(action_tokens.numel())
            completion_len = int(completion_ids.numel())
            has_action = 1.0 if action_len > 0 else 0.0

            self.log("probe_gen_action_candidate_count", float(action_candidates), sync_dist=False, on_step=True, on_epoch=False)
            self.log("probe_gen_action_tokens_len", float(action_len), sync_dist=False, on_step=True, on_epoch=False)
            self.log("probe_gen_has_action", float(has_action), sync_dist=False, on_step=True, on_epoch=False)
            self.log("probe_gen_completion_len", float(completion_len), sync_dist=False, on_step=True, on_epoch=False)
            self.log("probe_gen_prompt_len", float(prompt_len), sync_dist=False, on_step=True, on_epoch=False)
        except Exception as exc:
            print(f"[probe_gen] failed at step={step}: {type(exc).__name__}: {exc}")
            self.log("probe_gen_error", 1.0, sync_dist=False, on_step=True, on_epoch=False)
        finally:
            if was_training:
                vlm.train()

    def training_step(self, batch):
        hascot = batch['has_cot']
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        output = self.autovla(batch)
        base_loss = output.loss
        loss = base_loss

        # === Add additional loss on action tokens ===
        # output.logits shape: (B, T, V), labels shape: (B, T)
        logits = output.logits
        vocab_size = logits.size(-1)
        # Flatten logits and labels for token-wise loss
        labels = batch['labels']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        # Identify action token positions
        ignore_index = int(self.cfg['model']['tokens']['ignore_index'])
        valid_mask = (labels_flat != ignore_index)
        action_mask = valid_mask & (labels_flat >= self.autovla.action_start_id)  # shape: (B*T,)
        # Compute token-wise cross-entropy loss
        ce_loss_all = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none',
            ignore_index=ignore_index,
        )  # shape: (B*T,)
        # Extract loss for action tokens
        action_loss = ce_loss_all[action_mask]
        action_token_count = action_mask.sum()
        valid_token_count = valid_mask.sum()
        action_ratio = action_token_count.float() / valid_token_count.clamp_min(1).float()

        # Add action-focused loss for all samples (not only has_cot).
        if action_loss.numel() > 0:
            action_loss = action_loss.mean()
        else:
            action_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        action_loss_weight = float(self.cfg.get('training', {}).get('action_loss_weight', 1.0))
        loss = loss + action_loss_weight * action_loss

        # Keep legacy CoT multiplier configurable but disabled by default.
        cot_loss_multiplier = float(self.cfg.get('training', {}).get('cot_sample_loss_multiplier', 1.0))
        hascot_ratio = hascot.float().mean().to(loss.device)
        if cot_loss_multiplier != 1.0:
            loss = loss * (1.0 + (cot_loss_multiplier - 1.0) * hascot_ratio)

        self.log("train_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True,
                 prog_bar=True)
        self.log("train_base_loss", base_loss.detach(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_action_loss", action_loss.detach(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_action_loss_weight", action_loss_weight, sync_dist=False, on_step=True, on_epoch=False)
        self.log("train_action_token_count", action_token_count.float(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_valid_token_count", valid_token_count.float(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_action_token_ratio", action_ratio.detach(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_has_cot_ratio", hascot_ratio.detach(), sync_dist=True, on_step=True, on_epoch=False)
        self.log("train_cot_loss_multiplier", cot_loss_multiplier, sync_dist=False, on_step=True, on_epoch=False)
        self._maybe_log_generated_action_probe(batch)

        return loss
    
    def validation_step(self, batch):
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]

        output = self.autovla(batch)
        loss = output.loss
        self.log("val_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in get_vision_backbone(self.autovla.vlm).parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in get_language_backbone(self.autovla.vlm).parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training'].get('weight_decay', 0.0)
        )
        lr_warmpup_step = self.cfg['training']['lr_warmup_step']
        lr_step_freq = self.cfg['training']['lr_step_frequency']
        lr_step_gamma = self.cfg['training']['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def calculate_metrics(self, logits, labels, gt_trajectory):
        # Find start index for ground truth sequence
        gt_start_idx = self.find_assistant_start_idx(labels[0])
        gt_tokens = labels[0, gt_start_idx+1:] # shifted
        pred_tokens = logits[0, gt_start_idx:-1].argmax(dim=-1)

        # Find action tokens in ground truth and predicted sequences
        gt_action_idx = gt_tokens >= self.autovla.action_start_id
        pred_action_idx = pred_tokens >= self.autovla.action_start_id

        if len(pred_tokens[pred_action_idx]) != len(gt_tokens[gt_action_idx]):
            pred_action_idx = gt_action_idx
            
        gt_action_tokens = gt_tokens[gt_action_idx]
        pred_action_tokens = pred_tokens[pred_action_idx]

        # Decode predicted trajectory
        # pred_trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(pred_action_tokens.cpu())
        # action_acc = (pred_action_tokens == gt_action_tokens).float().mean()
        # traj_mse = torch.norm(pred_trajectory[0, 1:, :2] - gt_trajectory[0].cpu(), dim=-1).mean()
        # traj_mse = traj_mse.to(logits.device)

        # return {
        #     'action_acc': action_acc,
        #     'traj_mse': traj_mse
        # }
    
    @staticmethod
    def find_assistant_start_idx(labels):
        assistant_id = torch.tensor(ASSISTANT_ID).to(labels.device)
        
        for j in range(len(labels) - len(assistant_id) + 1):
            if torch.equal(labels[j:j + len(assistant_id)], assistant_id):
                start_idx = j
                break

        return start_idx


class AutoVLA(torch.nn.Module):
    def __init__(self, config, inference=False, device='cpu'):
        super().__init__()
        self.device = device

        model_path = config['model']['pretrained_model_path']
        self.model_family = detect_model_family(model_path)
        self.vlm = load_causal_lm_for_model(model_path, device=device)
        self.processor = load_processor_for_model(model_path)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, 
                                                model_config=config['model'])
        resize_token_embeddings_for_model(self.vlm, len(self.processor.tokenizer))
        initialize_model_runtime_state(self.vlm, self.processor)

        self.video_conf = config['model']['video']
        self.action_start_id = config['model']['tokens']['action_start_id']
        self._trajectory_num_poses = int(config['model']['trajectory']['num_poses'])
        self._action_answer_protocol_enabled = bool(
            config.get('model', {}).get('action_answer_protocol', {}).get('enabled', False)
        )

        self.use_cot = config['model']['use_cot']
        self.gen_conf = config['inference']['sample']
        self._inference_max_length = config.get('inference', {}).get('max_length', self.gen_conf.get('max_length'))
        self._last_protocol_result = {
            "protocol_valid": 0,
            "invalid_reason": "not_run",
            "answer_block_count": 0,
            "answer_block_at_tail": 0,
            "answer_action_tokens_len": 0,
        }

    def _vision_dtype(self) -> torch.dtype:
        vision_backbone = get_vision_backbone(self.vlm)
        first_param = next(vision_backbone.parameters(), None)
        if first_param is None:
            return torch.float32
        return first_param.dtype

    def _align_model_inputs(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return align_vision_tensor_dtypes(model_inputs, self._vision_dtype())

    def _build_qwen_inputs(self, messages):
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _build_internvl_inputs(self, messages):
        image_paths = []
        for message in messages:
            if message.get("role") != "user":
                continue
            for item in message.get("content", []):
                if item.get("type") == "video":
                    image_paths.extend(item.get("video", []))
                elif item.get("type") == "image":
                    image_paths.append(item.get("image"))
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor.build_batch(
            text=[text],
            image_paths=[image_paths],
            padding=True,
            return_tensors="pt",
        )

    def _parse_action_answer_completion(self, completion_ids: torch.Tensor):
        return parse_action_answer_completion(
            completion_ids.detach().cpu(),
            tokenizer=self.processor.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self._trajectory_num_poses,
        )

    def _set_last_protocol_result(self, parse_result=None) -> None:
        if parse_result is None:
            self._last_protocol_result = {
                "protocol_valid": 0,
                "invalid_reason": "not_run",
                "answer_block_count": 0,
                "answer_block_at_tail": 0,
                "answer_action_tokens_len": 0,
            }
            return
        self._last_protocol_result = {
            "protocol_valid": int(parse_result.is_valid),
            "invalid_reason": parse_result.invalid_reason,
            "answer_block_count": int(parse_result.answer_block_count),
            "answer_block_at_tail": int(parse_result.answer_block_at_tail),
            "answer_action_tokens_len": int(parse_result.action_token_count),
        }

    def _decode_protocol_trajectory(self, action_token_ids: List[int]) -> torch.Tensor:
        if not action_token_ids:
            return torch.zeros((0, 3), dtype=torch.float32)
        token_tensor = torch.tensor(action_token_ids, dtype=torch.long)
        decoded = self.action_tokenizer.decode_token_ids_to_trajectory(token_tensor)
        if isinstance(decoded, torch.Tensor) and decoded.ndim == 3 and decoded.shape[2] == 3 and decoded.shape[0] > 0:
            return decoded[0, 1:]
        return torch.zeros((0, 3), dtype=torch.float32)

    def _normalize_predicted_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.shape[0] == 0:
            return torch.zeros((self._trajectory_num_poses, 3), dtype=torch.float32)
        if trajectory.shape[0] < self._trajectory_num_poses:
            pad = trajectory[-1:].repeat(self._trajectory_num_poses - trajectory.shape[0], 1)
            return torch.cat([trajectory, pad], dim=0)
        return trajectory[: self._trajectory_num_poses]

    def predict(self, input_features):
        inputs = self.get_prompt(input_features)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        model_inputs = self._align_model_inputs(model_inputs)

        gen_kwargs = {
            "do_sample": True,
            "temperature": self.gen_conf['temperature'],
            "top_k": self.gen_conf['top_k'],
            "top_p": self.gen_conf['top_p'],
        }
        gen_kwargs.update(
            resolve_generate_length_kwargs(
                self.vlm,
                configured_max_new_tokens=self.gen_conf.get('max_new_tokens'),
                configured_max_length=self._inference_max_length,
            )
        )

        generate_inputs = filter_generate_inputs_for_model(self.vlm, model_inputs)
        outputs = self.vlm.generate(**generate_inputs, **gen_kwargs)

        input_ids = get_input_ids_tensor(inputs)
        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, outputs)
        ]

        outputs_trimmed = outputs_trimmed[0].cpu()
        cot_results = self.processor.decode(outputs_trimmed)
        if self._action_answer_protocol_enabled:
            parse_result = self._parse_action_answer_completion(outputs_trimmed)
            self._set_last_protocol_result(parse_result)
            if parse_result.is_valid:
                trajectory = self._decode_protocol_trajectory(parse_result.action_token_ids)
            else:
                trajectory = torch.zeros((0, 3), dtype=torch.float32)
        else:
            self._set_last_protocol_result(None)
            actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]
            decoded = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)
            if isinstance(decoded, torch.Tensor) and decoded.ndim == 3 and decoded.shape[2] == 3 and decoded.shape[0] > 0:
                trajectory = decoded[0, 1:]
            else:
                trajectory = torch.zeros((0, 3), dtype=torch.float32)
        trajectory = self._normalize_predicted_trajectory(trajectory)

        return trajectory, cot_results
    
    def get_prompt(self, input_features, image_mode="video"):
        # image sensor
        images = input_features['images']

        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        if input_features['sensor_data_path']:
            for camera_type in camera_types:
                camera_images[camera_type] = []
                for i in range(4):
                    img = images[camera_type][i]
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']


        # vehicle state
        velocity = input_features["vehicle_velocity"]

        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()
    
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next four seconds."
                )
            },
        ]

        if self.use_cot:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
                            "Your task is to predict the optimal driving action for the next four seconds.\n\n"
                            "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                            "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                            "Structure your reasoning as follows:\n"
                            "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                            "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                            "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                            "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                            "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                            "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                            "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                            "Present the final action clearly after your reasoning steps."
                        }
                    ]
                },

                {
                    "role": "user",
                    "content": user_content
                },


            ]
        else:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                            "Your task is to predict the most appropriate driving action for the next four seconds."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ]

        if self.model_family == "internvl_chat":
            return self._build_internvl_inputs(messages)
        return self._build_qwen_inputs(messages)
    
    def forward(self, inputs):
        inputs = dict(inputs)
        inputs.pop('gt_trajectory')
        inputs.pop('gt_action')
        inputs.pop('has_cot')
        inputs = self._align_model_inputs(inputs)
        outputs: CausalLMOutputWithPast = self.vlm(**inputs)

        return outputs
