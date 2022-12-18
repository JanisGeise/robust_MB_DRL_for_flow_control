"""Create a randomly initialized policy network.
"""
import sys
from os import environ
BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy


def create_dummy_policy(n_probes: int, cwd: str) -> None:
    policy = FCPolicy(n_probes, 1, -5.0, 5.0)
    script = pt.jit.script(policy)
    script.save(cwd + "/openfoam/test_cases/rotatingCylinder2D/policy.pt")


if __name__ == "__main__":
    create_dummy_policy(12, "..")
