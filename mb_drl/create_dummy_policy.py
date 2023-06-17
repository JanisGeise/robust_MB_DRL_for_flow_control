"""Create a randomly initialized policy network.
"""
import sys
from os import environ
from os.path import join
from typing import Union

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy


def create_dummy_policy(n_probes: int, cwd: str, abs_action: Union[int, float, pt.Tensor] = 5.0,
                        env: str = "rotatingCylinder2D") -> None:
    """
    initializes new policy, which is saved to '/openfoam/test_cases/rotatingCylinder2D/'
    :param n_probes: number of probes placed in the flow field
    :param cwd: current working directory of the training (should normally be path to drlfoam/)
    :param abs_action: absolute value of the action boundaries
    :param env: environment, eiter rotatingCylinder2D or rotatingPinball2D
    :return: None
    """
    policy = FCPolicy(n_probes, 1, -abs_action, abs_action)
    script = pt.jit.script(policy)
    script.save(join(cwd, "openfoam", "test_cases", env, "policy.pt"))


if __name__ == "__main__":
    create_dummy_policy(12, "..", 5.0, env="rotatingCylinder2D")
    create_dummy_policy(14, "..", 5.0, env="rotatingPinball2D")
