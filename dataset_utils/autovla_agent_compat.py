"""Compatibility loader for AutoVLAAgent across NavSim versions.

NavSim upstream v2 does not include AutoVLA's custom agent module.
This shim first tries the package import, and falls back to the local
repository implementation while keeping the rest of NavSim imports on v2.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_autovla_agent():
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    local_utils_path = project_root / "navsim" / "navsim" / "agents" / "utils.py"
    local_agent_path = project_root / "navsim" / "navsim" / "agents" / "autovla_agent.py"
    if not local_agent_path.exists() or not local_utils_path.exists():
        raise ModuleNotFoundError(
            "AutoVLAAgent fallback files are missing (autovla_agent.py or utils.py)."
        )

    # Inject local utility helpers into the v2 package namespace.
    if "navsim.agents.utils" not in sys.modules:
        _load_module("navsim.agents.utils", local_utils_path)

    module = _load_module("navsim.agents.autovla_agent", local_agent_path)
    return module.AutoVLAAgent


try:
    from navsim.agents.autovla_agent import AutoVLAAgent as _AutoVLAAgent
except ModuleNotFoundError:
    _AutoVLAAgent = _load_local_autovla_agent()


AutoVLAAgent = _AutoVLAAgent
