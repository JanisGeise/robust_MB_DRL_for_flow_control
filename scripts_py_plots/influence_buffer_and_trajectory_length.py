"""
    brief:
        - used for post-processing parameter study of PPO-Training routine wrt buffer size and trajectory length
        - reads in data of trainings and plots heatmaps wrt buffer size and trajectory length for cd, cl, run times and
          rewards

    dependencies:
        - 'post_process_ppo_results.py'
        - 'utils.py', located in the 'test_env_models' directory

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
from glob import glob
from typing import Union
from os import mkdir, path

import torch as pt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from test_env_models.utils import normalize_data
from post_process_ppo_results import average_results_for_each_case, load_all_data


def data_loader_wrapper_function(settings: dict) -> dict:
    """
    loads observations_*.pkl files of all cases of all subdirectories within a top-level directory, avg. each case over
    different seeds (if specified), then avg. every case over each episodes and then avg. over all episodes. So in the
    end for each case there exist only one mean and std. deviation for each parameter representing the complete training
    process

    :param settings: setup containing all the path etc.
    :return: dict containing the mean cl, cd, rewards and run times for each case (avg. over episodes) and the
             corresponding standard deviation
    """
    # get all directories containing data for the parameter study
    all_dirs = glob("".join([settings["main_load_path"], settings["path_controlled"], "/e*"]))

    # load the required run times from all cases
    run_times = pt.zeros((len(all_dirs), 2))
    for idx, case in enumerate(all_dirs):
        # get mean runtime in [s] of each case and corresponding standard deviation
        run_times[idx, :] = get_mean_run_time(case)

    # get the names of the case directories in order to sort the run times wrt to buffer size and trajectory length
    settings["case_name"] = ["".join([c.split("/")[-1], "/"]) for c in all_dirs]

    # load the training data for each case, avg. over different seeds and wrt to episode
    averaged_data = average_results_for_each_case(load_all_data(settings))

    # only the values for total mean & std are relevant
    out = {"mean_cl": averaged_data["tot_mean_cl"], "mean_cd": averaged_data["tot_mean_cd"],
           "mean_rewards": averaged_data["tot_mean_rewards"], "mean_runtime": run_times[:, 0],
           "std_cl": averaged_data["tot_std_cl"], "std_cd": averaged_data["tot_std_cd"],
           "std_rewards": averaged_data["tot_std_rewards"], "std_runtime": run_times[:, 1],
           "buffer_size": averaged_data["buffer_size"], "len_traj": averaged_data["len_traj"]
           }
    return out


def get_mean_run_time(path: str) -> pt.Tensor:
    """
    computes the mean duration of the PPO-training and the corresponding std. deviation using the log file

    :param path: path to the directory of the case containing the log files
    :return: list with mean and std. deviation of run time as [mean_run_time, std_dev_run_time]
    """
    # run time of training is saved at the end of each log_seed*.log file
    run_time = []
    for seed in glob("".join([path, "/log_seed*.log"])):
        with open(seed, "r") as f:
            run_time.append(float(f.readlines()[-1].split()[-1]))

    return pt.tensor((pt.mean(pt.tensor(run_time)).item(), pt.std(pt.tensor(run_time)).item()))


def map_data_to_tensor(data: dict, n_buffer: pt.Tensor, n_traj_len: pt.Tensor) -> list[list[pt.Tensor]]:
    """
    maps the loaded data to tensors, so they can be plotted as heatmaps, the data is loaded in arbitrary order, and
    therefore it needs to be determined which data set corresponds to which buffer_size-trajectory_length combination

    :param data: the loaded data containing all training results
    :param n_buffer: number of the different buffer sizes computed within the parameter study
    :param n_traj_len: number of the different trajectory lengths computed within the parameter study
    :return: list with mean and std. dev. as tensors as:
             [[mean_cl, std_cl], [mean_cd, std_cd], [mean_rewards, std_rewards], [mean_runtime, std_runtime],
              [mean(1 / rewards), std(1 / rewards)]]
    """
    # sort date wrt buffer size and trajectory length (rows = buffer size, cols = trajectory length)
    shape = (n_buffer.size()[0], n_traj_len.size()[0])
    mean_cl, mean_cd, mean_rewards, mean_runtime = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
    std_cl, std_cd, std_rewards, std_runtime = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)

    # create look-up table of buffer sizes and idx in heatmap, e.g. for b = 4, 10 -> b_idx = [(4, 0), (10, 1)] meaning
    # the buffer size of 4 is located in row 0 and buffer size of 10 is located in row 1
    b_idx = [(b.item(), i) for i, b in enumerate(n_buffer)]

    # do the same thing for the trajectory length, e.g. if l = 2, 4 -> l_idx = [(2, 0), (4, 1)] meaning the 2 sec long
    # trajectory is located in column 0 and the 4 sec long trajectory is located in column 1
    l_idx = [(l.item(), i) for i, l in enumerate(n_traj_len)]

    # then determine in which order the cases are loaded and create (buffer_size, traj_len) tuples from it
    order_loaded_b_l = [(b, l) for b, l in zip(data["buffer_size"], data["len_traj"])]

    # map the (buffer_size, traj_len) tuples to correct position within tensor for creating the heatmaps
    look_up_dict = {}
    for row in range(shape[0]):
        for col in range(shape[1]):
            look_up_dict[(b_idx[row][0], l_idx[col][0])] = (row, col)

    # replace the (buffer_size, traj_len) tuples with (idx_buffer, idx_len) tuples depending on their position in the
    # heatmap tensor
    idx_list = [*map(look_up_dict.get, order_loaded_b_l)]

    # sort in the loaded data to the tensors based on the idx_list, take abs() from cl
    for i, _ in enumerate(idx_list):
        mean_cl[idx_list[i]], mean_cd[idx_list[i]] = pt.abs(data["mean_cl"][i]), data["mean_cd"][i]
        mean_rewards[idx_list[i]], mean_runtime[idx_list[i]] = data["mean_rewards"][i], data["mean_runtime"][i]
        std_cl[idx_list[i]], std_cd[idx_list[i]] = data["std_cl"][i], data["std_cd"][i]
        std_rewards[idx_list[i]], std_runtime[idx_list[i]] = data["std_rewards"][i], data["std_runtime"][i]

    # scale everything to intervall [0, 1], use inverted rewards to calculate optimum since goal is to min(1/r, t)
    mean_1_rewards = normalize_data((1 / mean_rewards))[0]
    std_1_rewards = normalize_data(1 / std_rewards)[0]

    mean_cl = normalize_data(mean_cl)[0]
    std_cl = normalize_data(std_cl)[0]

    mean_cd = normalize_data(mean_cd)[0]
    std_cd = normalize_data(std_cd)[0]

    mean_rewards = normalize_data(mean_rewards)[0]
    std_rewards = normalize_data(std_rewards)[0]

    mean_runtime = normalize_data(mean_runtime)[0]
    std_runtime = normalize_data(std_runtime)[0]

    return [[mean_cl, std_cl], [mean_cd, std_cd], [mean_rewards, std_rewards], [mean_runtime, std_runtime],
            [mean_1_rewards, std_1_rewards]]


def plot_heatmaps(settings: dict, mean_data: Union[list, pt.Tensor], std_data: Union[list, pt.Tensor],
                  parameter: str = "c_d ") -> None:
    """
    :brief: creates heatmaps wrt buffer size and trajectory length for a specified parameter
            note:
                - seaborn plots row-wise (in x-direction) starting in the top-left corner
                    -> x-data is n_layers, y-data is n_neurons
    :param settings: setup defining paths etc.
    :param mean_data: tensor containing the mean values of all buffer size - trajectory length combinations
    :param std_data: tensor containing the corresponding std. dev. of all buffer size - trajectory length combinations
    :param parameter: name of the parameter which is plotted (only used for title and save name)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(2):
        if i == 0:
            if parameter == "c_l":
                ax[i].set_title(f"$\\mu(|{parameter}^*|)$ $\in$ $[0, 1]$", usetex=True, fontsize=16)
            elif parameter == "r_t":
                ax[i].set_title("$\\mu\left(\sum\left[\\frac{1}{r^*}, t^*\\right]\\right)$ $\in$ $[0, 1]$",
                                usetex=True, fontsize=16)
            else:
                ax[i].set_title(f"$\\mu({parameter}^*)$ $\in$ $[0, 1]$", usetex=True, fontsize=16)
            z, vmax, fmt = mean_data, 1, ".1f"
        else:
            if parameter == "c_l":
                ax[i].set_title(f"$\sigma(|{parameter}^*|)$ $\in$ $[0, 1]$", usetex=True, fontsize=16)
            elif parameter == "r_t":
                ax[i].set_title("$\sum\left\{\\mu\left(\sum\left[\\frac{1}{r^*}, t^*\\right]\\right),"
                                "\\sigma\left(\sum\left[\\frac{1}{r^*}, t^*\\right]\\right)\\right\}$ $\in$ $[0, 1]$",
                                usetex=True, fontsize=16)

                # draw rectangle around minimum
                row_idx = (pt.argmin(std_data) % std_data.size()[0]).item()
                col_idx = pt.argmin(std_data[row_idx], dim=0).item()
                patch = Rectangle((row_idx, col_idx), width=1, height=1, edgecolor="red", linewidth=2, facecolor="none")
                ax[i].add_patch(patch)
            else:
                ax[i].set_title(f"$\sigma({parameter}^*)$ $\in$ $[0, 1]$", usetex=True, fontsize=16)
            z, vmax, fmt = std_data, pt.max(std_data), ".2f"

        heatmap = sns.heatmap(z, vmin=0, vmax=vmax, center=0, cmap="Greens", square=True, annot=True, cbar=True,
                              linewidths=0.30, linecolor="white", ax=ax[i], xticklabels=settings["traj_len"],
                              yticklabels=settings["buffer"], cbar_kws={"shrink": 0.75}, fmt=fmt)
        if i == 1 and parameter == "r_t":
            heatmap.axes.add_patch(patch)
        ax[i].set_ylabel("$buffer$ $size$ $[-]$", usetex=True, fontsize=13, labelpad=15)
        ax[i].set_xlabel("$trajectory$ $length$ $[s]$", usetex=True, fontsize=13, labelpad=15)

        # since seaborn starts plotting as top-left corner -> axis needs to be inverted
        ax[i].invert_yaxis()
    fig.subplots_adjust(wspace=0.2)
    fig.tight_layout()
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"],
                         f"/plots/heatmaps/mean_std_{parameter}_vs_buffer_and_len_traj.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/robust_MB_DRL_for_flow_control/",    # top-level dir
        "path_controlled": r"run/influence_buffer_len_traj/",    # top-level dir for parameter study
        "avg_over_cases": True,                                # if cases should be averaged over, e.g. different seeds
    }

    # load all the data generated for the parameter study in a random order (gets wrt buffer & traj_len sorted later)
    loaded_data = data_loader_wrapper_function(setup)

    # determine how many buffer sizes and trajectory lengths are present in the data set
    n_buffer_size = pt.unique(pt.tensor(loaded_data["buffer_size"]))
    n_traj_length = pt.unique(pt.tensor(loaded_data["len_traj"]))

    # sort the loaded data into tensors wrt the buffer size & trajectory length, so they can be plotted as heatmaps
    cl, cd, rewards, runtime, inverse_rewards = map_data_to_tensor(loaded_data, n_buffer=n_buffer_size,
                                                                   n_traj_len=n_traj_length)

    # create directory for plots
    if not path.exists("".join([setup["main_load_path"], setup["path_controlled"], "plots/heatmaps"])):
        mkdir("".join([setup["main_load_path"], setup["path_controlled"], "plots/heatmaps"]))

    # plot results wrt trajectory length and buffer size as heatmaps
    setup["buffer"] = n_buffer_size.tolist()
    setup["traj_len"] = n_traj_length.tolist()

    plot_heatmaps(setup, mean_data=cl[0], std_data=cl[1], parameter="c_l")
    plot_heatmaps(setup, mean_data=cd[0], std_data=cd[1], parameter="c_d")
    plot_heatmaps(setup, mean_data=rewards[0], std_data=rewards[1], parameter="r")
    plot_heatmaps(setup, mean_data=runtime[0], std_data=runtime[1], parameter="t")
    plot_heatmaps(setup, mean_data=(inverse_rewards[0] + runtime[0]) / 2,
                  std_data=(inverse_rewards[0] + inverse_rewards[1] + runtime[1] + runtime[0]) / 4, parameter="r_t")
