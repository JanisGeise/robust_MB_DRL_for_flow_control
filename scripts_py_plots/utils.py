"""
    brief:
        functions for:
            - loading and splitting trajectory data
            - normalize the loaded data

    dependencies:
        None

    prerequisites:
        None
"""
import pickle
import psutil
import torch as pt

from glob import glob
from typing import Tuple
from natsort import natsorted


def load_trajectory_data(path: str, preserve_episodes: bool = False, len_traj: int = 400) -> dict:
    """
    load observations_*.pkl files containing all the data generated during training and sort them into a dict

    :param path: path to directory containing the files
    :param preserve_episodes: either 'True' if the data should be sorted wrt the episodes, or 'False' if the order of
                              the episodes doesn't matter (in case only one model is trained for all the data)
    :param len_traj: length of the trajectories defined in the setup, the loaded trajectories are split wrt this length
    :return: actions, states, cl, cd as tensors within a dict where every column is a trajectory, also return number of
             workers used for sampling the trajectories. The structure is either:
             - all trajectories of each parameter in one tensor, independent of the episode. The resulting length
               of the trajectories corresponds to the length defined in the setup. This is the case if
               'preserve_episodes = False'

             - each parameter contains one tensor with the length of N_episodes. Each entry contains all the
               trajectories sampled in this episode (columns = N_trajectories, rows = length_trajectories).
               This is the case if 'preserve_episodes = True'
    """
    # load the 'observations_*.pkl' files containing the trajectories sampled in the CFD environment
    files = natsorted(glob(path + "observations_*.pkl"))
    observations = [pickle.load(open(file, "rb")) for file in files]
    actual_traj_length = len(observations[0][0]["actions"])

    # make sure there are no invalid settings defined
    assert actual_traj_length % len_traj == 0, f"(trajectory length = {actual_traj_length}) % (len_trajectory =" \
                                               f"{len_traj}) != 0 "
    assert actual_traj_length >= len_traj, f"imported trajectories can't be extended from {actual_traj_length} to" \
                                           f" {len_traj}!"

    data = {"n_workers": len(observations[0])}

    # if training is episode wise: sort the trajectories from all workers wrt the episode
    if preserve_episodes:
        factor = len_traj / actual_traj_length
        shape = (actual_traj_length, data["n_workers"])
        actions, cl, cd = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
        states = pt.zeros((shape[0], observations[0][0]["states"].size()[1], shape[1]))
        shape = (len(observations), int(actual_traj_length * factor), int(data["n_workers"] / factor))
        data["actions"], data["cl"], data["cd"] = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
        data["states"] = pt.zeros((shape[0], shape[1], observations[0][0]["states"].size()[1], shape[2]))
        for episode in range(len(observations)):
            for worker in range(len(observations[episode])):
                actions[:, worker] = observations[episode][worker]["actions"]
                states[:, :, worker] = observations[episode][worker]["states"]
                cl[:, worker] = observations[episode][worker]["cl"]
                cd[:, worker] = observations[episode][worker]["cd"]
            data["actions"][episode, :, :] = pt.concat(pt.split(actions, len_traj), dim=1)
            data["states"][episode, :, :] = pt.concat(pt.split(states, len_traj), dim=2)
            data["cl"][episode, :, :] = pt.concat(pt.split(cl, len_traj), dim=1)
            data["cd"][episode, :, :] = pt.concat(pt.split(cd, len_traj), dim=1)

    # if only one model is trained using all available data, the order of the episodes doesn't matter
    else:
        shape = (actual_traj_length, len(observations) * len(observations[0]))
        n_probes, n_col = len(observations[0][0]["states"][0]), 0
        states = pt.zeros((shape[0], n_probes, shape[1]))
        actions, cl, cd = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)

        for observation in range(len(observations)):
            for j in range(len(observations[observation])):
                actions[:, n_col] = observations[observation][j]["actions"]
                cl[:, n_col] = observations[observation][j]["cl"]
                cd[:, n_col] = observations[observation][j]["cd"]
                states[:, :, n_col] = observations[observation][j]["states"][:]
                n_col += 1

        data["actions"] = pt.concat(pt.split(actions, len_traj), dim=1)
        data["cl"] = pt.concat(pt.split(cl, len_traj), dim=1)
        data["cd"] = pt.concat(pt.split(cd, len_traj), dim=1)
        data["states"] = pt.concat(pt.split(states, len_traj), dim=2)

    return data


def split_data(states: pt.Tensor, actions: pt.Tensor, cl: pt.Tensor, cd: pt.Tensor, ratio: Tuple) -> dict:
    """
    split trajectories into train-, validation and test data

    :param states: sampled states in CFD environment
    :param actions: sampled actions in CFD environment
    :param cd: sampled cl-coefficients in CFD environment (at cylinder surface)
    :param cl: sampled cd-coefficients in CFD environment (at cylinder surface)
    :param ratio: ratio between training-, validation- and test data as tuple as (train, validation, test)
    :return: dictionary with splitted states and corresponding actions
    """
    data = {}
    # split dataset into training data, validation data and testdata
    if ratio[2] == 0:
        n_train, n_test = int(ratio[0] * actions.size()[1]), 0
        n_val = actions.size()[1] - n_train
    else:
        n_train, n_val = int(ratio[0] * actions.size()[1]), int(ratio[1] * actions.size()[1])
        n_test = actions.size()[1] - n_train - n_val

    # randomly select indices of trajectories
    samples = pt.ones(actions.shape[-1])
    idx_train = pt.multinomial(samples, n_train)
    idx_val = pt.multinomial(samples, n_val)

    # assign train-, validation and testing data based on chosen indices
    data["actions_train"], data["actions_val"] = actions[:, idx_train], actions[:, idx_val]
    data["states_train"], data["states_val"] = states[:, :, idx_train], states[:, :, idx_val]
    data["cl_train"], data["cd_train"] = cl[:, idx_train], cd[:, idx_train]
    data["cl_val"], data["cd_val"] = cl[:, idx_val], cd[:, idx_val]

    # if predictions are independent of episode: split test data, otherwise test data are the trajectories of the next
    # episode (take N episodes for training and try to predict episode N+1)
    if n_test != 0:
        idx_test = pt.multinomial(samples, n_test)
        data["cl_test"], data["cd_test"] = cl[:, idx_test], cd[:, idx_test]
        data["states_test"], data["actions_test"] = states[:, :, idx_test], actions[:, idx_test]

    return data


def normalize_data(x: pt.Tensor) -> Tuple[pt.Tensor, list]:
    """
    normalize data to the interval [0, 1] using a min-max-normalization

    :param x: data which should be normalized
    :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
    """
    # x_i_normalized = (x_i - x_min) / (x_max - x_min)
    x_min_max = [pt.min(x), pt.max(x)]
    return pt.sub(x, x_min_max[0]) / (x_min_max[1] - x_min_max[0]), x_min_max


def print_core_temp():
    """
    prints the current processor temperature of all available cores for monitoring

    :return: None
    """
    temps = psutil.sensors_temperatures()["coretemp"]
    print(f"{(4 * len(temps) + 1) * '-'}\n\tcore number\t|\ttemperature [deg]\t \n{(4 * len(temps) + 1) * '-'}")
    for i in range(len(temps)):
        print(f"\t\t{i}\t\t|\t{temps[i][1]}")


def dataloader_wrapper(settings: dict) -> dict:
    """
    load trajectory data, normalizes and splits the data into training-, validation- and testing data

    :param settings: setup defining paths, splitting rations etc.
    :return: dict containing all the trajectory data required for train, validate and testing the environment model
    """
    all_data = load_trajectory_data(settings["load_path"], settings["episode_depending_model"],
                                    settings["len_trajectory"])

    if not settings["episode_depending_model"]:
        print(f"data contains {all_data['actions'].size()[-1]} trajectories with length of"
              f" {all_data['actions'].size()[0]} entries per trajectory")
        all_data["n_probes"] = all_data["states"].size()[1]
    else:
        all_data["n_probes"] = all_data["states"][0].size()[1]

    # normalize the data
    if settings["normalize"]:
        all_data["actions"], all_data["min_max_actions"] = normalize_data(all_data["actions"])
        all_data["cl"], all_data["min_max_cl"] = normalize_data(all_data["cl"])
        all_data["cd"], all_data["min_max_cd"] = normalize_data(all_data["cd"])
        all_data["states"], all_data["min_max_states"] = normalize_data(all_data["states"])

    # split dataset into training-, validation- and test data if whole data set used for train only one (global) model
    if not settings["episode_depending_model"]:
        all_data.update(split_data(all_data["states"], all_data["actions"], all_data["cl"], all_data["cd"],
                                   ratio=settings["ratio"]))
        del all_data["actions"], all_data["states"], all_data["cl"], all_data["cd"]

    # load value-, policy and MSE losses of PPO training
    all_data["network_data"] = pickle.load(open(settings["load_path"] + "training_history.pkl", "rb"))

    return all_data


if __name__ == "__main__":
    pass
