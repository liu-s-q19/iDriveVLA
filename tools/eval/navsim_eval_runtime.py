import random
from hashlib import sha256
from typing import Iterable, List, Mapping, Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def intersect_tokens_preserve_order(scene_tokens: Iterable[str], metric_tokens: Iterable[str]) -> List[str]:
    metric_token_set = set(metric_tokens)
    return [token for token in scene_tokens if token in metric_token_set]


def build_generation_kwargs(sample_conf: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "max_length": sample_conf["max_length"],
        "do_sample": sample_conf.get("do_sample", True),
        "temperature": sample_conf["temperature"],
        "top_k": sample_conf["top_k"],
        "top_p": sample_conf["top_p"],
    }


def token_seed(base_seed: int, token: str) -> int:
    token_hash = sha256(token.encode("utf-8")).digest()
    token_offset = int.from_bytes(token_hash[:8], byteorder="big", signed=False)
    return (base_seed + token_offset) % (2**32)
