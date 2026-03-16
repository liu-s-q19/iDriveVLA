import inspect


def init_abstract_agent_compat(agent, init_fn, trajectory_sampling, requires_scene=False):
    """Call AbstractAgent.__init__ across old/new NavSim signatures."""
    params = inspect.signature(init_fn).parameters
    if "trajectory_sampling" in params:
        return init_fn(agent, trajectory_sampling=trajectory_sampling, requires_scene=requires_scene)
    return init_fn(agent, requires_scene=requires_scene)
