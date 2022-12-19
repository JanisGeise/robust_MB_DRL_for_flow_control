"""
    get the number of probes defined in the control dict of the base case, then initialize a dummy policy and the
    RotatingCylinder class
"""
import re

from create_dummy_policy import create_dummy_policy


def get_number_of_probes(cwd) -> int:
    """
    get the number of probes defined in the control dict of the base case, initialize a dummy policy with N_probes

    :param: current working directory, expected to be "drlfoam/"
    :return: N_probes defined in the control dict of the base case
    """
    # in case drlfoam is  run on HPC / using container, the path is different, so remove the last directory from it
    if cwd.endswith("examples"):
        cwd = "".join(cwd.split("/")[:-1])
    path_to_dict = cwd + r"/openfoam/test_cases/rotatingCylinder2D/system/controlDict"
    key = "probeLocations"
    with open(path_to_dict, "rb") as f:
        lines = f.readlines()

    # get dict containing all probe locations
    lines = [l.decode("utf-8") for l in lines]
    start = [(re.findall(key, l), idx) for idx, l in enumerate(lines) if re.findall(key, l)][0][1]
    end = [(re.findall("\);", l), idx) for idx, l in enumerate(lines) if re.findall("\);", l) and idx > start][0][1]

    # strip everything but the probe locations, start + 2 because "probeLocations" and "(\n" are in list
    lines = lines[start+2:end]

    # create a dummy policy
    create_dummy_policy(len(lines), cwd)

    return len(lines)


if __name__ == "__main__":
    pass
