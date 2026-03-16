import unittest

from navsim_ext.agent_ctor_compat import init_abstract_agent_compat


class DummyAgent:
    pass


class TestAgentCtorCompat(unittest.TestCase):
    def test_prefers_new_signature_without_trajectory_sampling(self):
        calls = []

        def init_fn(self, requires_scene=False):
            calls.append((self, requires_scene))

        agent = DummyAgent()
        init_abstract_agent_compat(agent, init_fn, trajectory_sampling="traj", requires_scene=False)

        self.assertEqual(calls, [(agent, False)])

    def test_falls_back_to_old_signature_with_trajectory_sampling(self):
        calls = []

        def init_fn(self, trajectory_sampling, requires_scene=False):
            calls.append((self, trajectory_sampling, requires_scene))

        agent = DummyAgent()
        init_abstract_agent_compat(agent, init_fn, trajectory_sampling="traj", requires_scene=False)

        self.assertEqual(calls, [(agent, "traj", False)])


if __name__ == "__main__":
    unittest.main()
