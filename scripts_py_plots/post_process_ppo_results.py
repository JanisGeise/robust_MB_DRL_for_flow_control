"""
    brief:
        - post-processes and plot the result of the training using PPO and CFD as an environment
        - plots the results of the controlled case using the best policy in comparison to the uncontrolled case

    dependencies:
        - 'analyze_frequency_spectrum.py' for plotting the frequency spectrum of cl- and cd of PPO-training and the
           final results

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to conduct a training
          (https://github.com/OFDataCommittee/drlfoam)
        - execution of simulation for the best policy from training, also results of a simulation without control
"""
import os
import re
import pickle
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from typing import Union
from natsort import natsorted
from matplotlib.patches import Circle, Rectangle

from analyze_frequency_spectrum import analyze_frequencies_final_result, analyze_frequencies_ppo_training


def load_trajectory_data(path: str) -> dict:
    """
    load observations_*.pkl files containing all the data generated during training and sort them into a dict

    :param path: path to directory containing the files
    :return: dict with actions, states, cl, cd. Each parameter contains one tensor with the length of N_episodes, each
             entry has all the trajectories sampled in this episode (cols = N_trajectories, rows = length_trajectories)
    """
    # sort imported data wrt to episode number
    files = natsorted(glob(path + "observations_*.pkl"))
    observations = [pickle.load(open(file, "rb")) for file in files]
    traj_length = len(observations[0][0]["actions"])

    data = {"n_workers": len(observations[0])}

    # sort the trajectories from all workers wrt the episode
    shape, counter = (traj_length, data["n_workers"]), 0
    actions, cl, cd, rewards, alpha, beta = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), \
                                            pt.zeros(shape), pt.zeros(shape)
    states = pt.zeros((shape[0], observations[0][0]["states"].size()[1], shape[1]))
    shape = (len(observations), traj_length, data["n_workers"])
    data["actions"], data["cl"], data["cd"], = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
    data["rewards"], data["alpha"], data["beta"], = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
    data["states"] = pt.zeros((shape[0], shape[1], observations[0][0]["states"].size()[1], shape[2]))
    for episode in range(len(observations)):
        for worker in range(len(observations[episode])):
            # omit failed trajectories
            if observations[episode][worker]["actions"].size()[0] < traj_length:
                counter += 1
                continue
            else:
                actions[:, worker] = observations[episode][worker]["actions"]
                states[:, :, worker] = observations[episode][worker]["states"]
                cl[:, worker] = observations[episode][worker]["cl"]
                cd[:, worker] = observations[episode][worker]["cd"]
                rewards[:, worker] = observations[episode][worker]["rewards"]
                alpha[:, worker] = observations[episode][worker]["alpha"]
                beta[:, worker] = observations[episode][worker]["beta"]

        data["actions"][episode, :, :] = actions
        data["states"][episode, :, :] = states
        data["cl"][episode, :, :] = cl
        data["cd"][episode, :, :] = cd
        data["rewards"][episode, :, :] = rewards
        data["alpha"][episode, :, :] = alpha
        data["beta"][episode, :, :] = beta

    # check how many trajectories failed
    if counter > 0:
        print(f"found {counter} failed trajectories")

    # load value-, policy and MSE losses of PPO training
    data["network_data"] = pickle.load(open(path + "training_history.pkl", "rb"))

    return data


def load_all_data(settings: dict) -> list[dict]:
    """
    wrapper function for loading the results of the PPO-training and sorting it wrt episodes

    :param settings: setup containing all paths etc.
    :return: a list containing a dictionary with all the data from each case
    """
    # load the results of the training
    loaded_data = []

    if settings["avg_over_cases"]:
        for c in range(len(settings["case_name"])):
            case_data = []

            # assuming each case directory contains subdirectories with training data ran with different seeds,
            # exclude directories named "plots", readme files, logs etc.
            dirs = [d for d in glob("".join([settings["main_load_path"], settings["path_controlled"],
                                             settings["case_name"][c], "/seed[0-9]"]))]

            for d in dirs:
                case_data.append(load_trajectory_data(d + "/"))

            # merge training results from same case, but different seeds episode-wise
            loaded_data.append(merge_results_for_diff_seeds(case_data, n_seeds=len(case_data)))

    else:
        for c in range(len(settings["case_name"])):
            loaded_data.append(load_trajectory_data(settings["main_load_path"] + settings["path_controlled"] +
                                                    settings["case_name"][c]))
    return loaded_data


def average_results_for_each_case(data: list) -> dict:
    """
    average the loaded results of the training periode wrt the episode for each imported case

    :param data: list containing a dict for each imported case containing all the results of the training
    :return: dict containing the mean actions, cl, cd, alpha, beta and rewards as well as the corresponding standard
             deviation wrt the episode
    """
    # calculate the mean and std. deviation in each episode for each case
    avg_data = {"mean_cl": [], "std_cl": [], "mean_cd": [], "std_cd": [], "mean_actions": [], "std_actions": [],
                "mean_rewards": [], "std_rewards": [], "mean_alpha": [], "std_alpha": [], "mean_beta": [],
                "std_beta": [], "tot_mean_rewards": [], "tot_std_rewards": [], "tot_mean_cd": [], "tot_std_cd": [],
                "tot_mean_cl": [], "tot_std_cl": [], "var_beta_fct": [], "buffer_size": [], "len_traj": []}

    for case in range(len(data)):
        n_episodes, len_trajectory = data[case]["actions"].size()[0], data[case]["actions"].size()[1]

        # calculate avg. and std. dev. of all trajectories within episode
        cl_reshaped = data[case]["cl"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
        cd_reshaped = data[case]["cd"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
        actions_reshaped = data[case]["actions"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
        reward_reshaped = data[case]["rewards"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
        alpha_reshaped = data[case]["alpha"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
        beta_reshaped = data[case]["beta"].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))

        avg_data["mean_cl"].append(pt.mean(cl_reshaped, dim=1))
        avg_data["mean_cd"].append(pt.mean(cd_reshaped, dim=1))
        avg_data["mean_actions"].append(pt.mean(actions_reshaped, dim=1))
        avg_data["mean_rewards"].append(pt.mean(reward_reshaped, dim=1))
        avg_data["mean_alpha"].append(pt.mean(alpha_reshaped, dim=1))
        avg_data["mean_beta"].append(pt.mean(beta_reshaped, dim=1))

        avg_data["std_cl"].append(pt.std(cl_reshaped, dim=1))
        avg_data["std_cd"].append(pt.std(cd_reshaped, dim=1))
        avg_data["std_actions"].append(pt.std(actions_reshaped, dim=1))
        avg_data["std_rewards"].append(pt.std(reward_reshaped, dim=1))
        avg_data["std_alpha"].append(pt.std(alpha_reshaped, dim=1))
        avg_data["std_beta"].append(pt.std(beta_reshaped, dim=1))

        # compute variance of the (mean) beta-distribution of each episode
        # var = (alpha*beta) / ((alpha + beta)^2 * (alpha+beta+1))
        var = (avg_data["mean_alpha"][case] * avg_data["mean_beta"][case]) / \
              ((avg_data["mean_alpha"][case] + avg_data["mean_beta"][case]) ** 2 *
               (avg_data["mean_alpha"][case] + avg_data["mean_beta"][case] + 1))
        avg_data["var_beta_fct"].append(var)

        # total rewards, cl and cd of complete training for each case
        avg_data["tot_mean_rewards"].append(pt.mean(data[case]["rewards"]))
        avg_data["tot_std_rewards"].append(pt.std(data[case]["rewards"]))
        avg_data["tot_mean_cl"].append(pt.mean(data[case]["cl"]))
        avg_data["tot_std_cl"].append(pt.std(data[case]["cl"]))
        avg_data["tot_mean_cd"].append(pt.mean(data[case]["cd"]))
        avg_data["tot_std_cd"].append(pt.std(data[case]["cd"]))

        # info about the setup, assuming constant sample rate of 100 Hz
        if "n_seeds" in data[case]:
            avg_data["buffer_size"].append(int(data[case]["n_workers"] / data[case]["n_seeds"]))

        else:
            avg_data["buffer_size"].append(data[case]["n_workers"])
        avg_data["len_traj"].append(int(len_trajectory / 100))

    return avg_data


def plot_results_vs_episode(settings: dict, cd_mean: Union[list, pt.Tensor], cd_std: Union[list, pt.Tensor],
                            cl_mean: Union[list, pt.Tensor], cl_std: Union[list, pt.Tensor],
                            actions_mean: Union[list, pt.Tensor], actions_std: Union[list, pt.Tensor],
                            n_cases: int = 1, plot_action: bool = True) -> None:
    """
    plot cl, cd and actions (if specified) depending on the episode (training)

    :param settings: dict containing all the paths etc.
    :param cd_mean: mean cd received over the training periode
    :param cd_std: corresponding standard deviation of cd throughout the training periode
    :param cl_mean: mean cl received over the training periode
    :param cl_std: corresponding standard deviation of cl throughout the training periode
    :param actions_mean: mean actions (omega) done over the training periode
    :param actions_std: corresponding standard deviation of the actions done over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :param plot_action: if 'True' cl, cd and actions will be plotted, otherwise only cl and cd will be plotted
    :return: None
    """
    if plot_action:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
        n_subfig = 3
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        n_subfig = 2

    for c in range(n_cases):
        for i in range(n_subfig):
            if i == 0:
                ax[i].plot(range(len(cl_mean[c])), cl_mean[c], color=settings["color"][c], label=settings["legend"][c])
                ax[i].fill_between(range(len(cl_mean[c])), cl_mean[c] - cl_std[c], cl_mean[c] + cl_std[c],
                                   color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$mean$ $lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)

            elif i == 1:
                ax[i].plot(range(len(cd_mean[c])), cd_mean[c], color=settings["color"][c])
                ax[i].fill_between(range(len(cd_mean[c])), cd_mean[c] - cd_std[c], cd_mean[c] + cd_std[c],
                                   color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$mean$ $drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)

            elif plot_action:
                ax[i].plot(range(len(actions_mean[c])), actions_mean[c], color=settings["color"][c])
                ax[i].fill_between(range(len(actions_mean[c])), actions_mean[c] - actions_std[c],
                                   actions_mean[c] + actions_std[c], color=settings["color"][c], alpha=0.3)
                ax[i].set_ylabel("$mean$ $action$ $\qquad \omega$", usetex=True, fontsize=13)

            ax[i].set_xlabel("$episode$ $number$", usetex=True, fontsize=13)

    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=n_cases)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig("".join([settings["main_load_path"], setup["path_controlled"], "/plots/coefficients_vs_episode.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_rewards_vs_episode(settings: dict, reward_mean: Union[list, pt.Tensor], reward_std: Union[list, pt.Tensor],
                            n_cases: int = 0) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param settings: dict containing all the paths etc.
    :param reward_mean: mean rewards received over the training periode
    :param reward_std: corresponding standard deviation of the rewards received over the training periode
    :param n_cases: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    for c in range(n_cases):
        ax.plot(range(len(reward_mean[c])), reward_mean[c], color=settings["color"][c], label=settings["legend"][c])
        ax.fill_between(range(len(reward_mean[c])), reward_mean[c] - reward_std[c], reward_mean[c] + reward_std[c],
                        color=settings["color"][c], alpha=0.3)

    ax.set_ylabel("$mean$ $reward$", usetex=True, fontsize=12)
    ax.set_xlabel("$episode$ $number$", usetex=True, fontsize=12)
    ax.legend(loc="lower right", framealpha=1.0, fontsize=10, ncol=2)
    plt.savefig("".join([settings["main_load_path"], setup["path_controlled"], "/plots/rewards_vs_episode.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_alpha_beta(settings: dict, controlled_cases: Union[list, pt.Tensor],
                          uncontrolled_case: Union[list, pt.Tensor] = None, plot_coeffs=True) -> None:
    """
    plot either cl and cd vs. time or alpha and beta vs. time

    :param settings: dict containing all the paths etc.
    :param controlled_cases: results from the loaded cases with active flow control
    :param uncontrolled_case: reference case containing results from uncontrolled flow past cylinder
    :param plot_coeffs: 'True' means cl and cd will be plotted, otherwise alpha and beta will be plotted wrt to time
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    if plot_coeffs:
        keys = ["t", "cl", "cd"]
        save_name = "comparison_cl_cd"
        ax[1].set_ylim(2.95, 3.25)
        n_cases = range(len(settings["case_name"]) + 1)
        ylabels = ["$lift$ $coefficient$ $\qquad c_l$", "$drag$ $coefficient$ $\qquad c_d$"]
    else:
        keys = ["t", "alpha", "beta"]
        save_name = "comparison_alpha_beta"
        ylabels = ["$\\alpha$", "$\\beta$"]
        n_cases = range(1, len(settings["case_name"]) + 1)

    for c in n_cases:
        for i in range(2):
            if i == 0:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]], uncontrolled_case[keys[1]], color="black",
                               label="uncontrolled")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]], controlled_cases[c - 1][keys[1]],
                               color=setup["color"][c - 1], label=settings["legend"][c - 1])
                ax[i].set_ylabel(ylabels[0], usetex=True, fontsize=13)
            else:
                if c == 0:
                    ax[i].plot(uncontrolled_case[keys[0]], uncontrolled_case[keys[2]], color="black")
                else:
                    ax[i].plot(controlled_cases[c - 1][keys[0]], controlled_cases[c - 1][keys[2]],
                               color=settings["color"][c - 1])
                ax[i].set_ylabel(ylabels[1], usetex=True, fontsize=13)

            ax[i].set_xlabel("$time$ $[s]$", usetex=True, fontsize=13)
    fig.suptitle("", usetex=True, fontsize=14)
    fig.tight_layout(pad=1.5)
    fig.legend(loc="upper right", framealpha=1.0, fontsize=11, ncol=len(settings["case_name"]) + 1)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"], f"/plots/{save_name}.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_omega(settings: dict, controlled_cases: Union[list, pt.Tensor]) -> None:
    """
    plot omega (actions) vs. time

    :param settings: dict containing all the paths etc.
    :param controlled_cases: results from the loaded cases with active flow control
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for c in range(len(settings["case_name"])):
        ax.plot(controlled_cases[c]["t"], controlled_cases[c]["omega"], color=settings["color"][c],
                label=settings["legend"][c])

    ax.set_ylabel("$action$ $\omega$", usetex=True, fontsize=13)
    ax.set_xlabel("$time$ $[s]$", usetex=True, fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=len(settings["case_name"]))
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"], f"/plots/omega_controlled_case.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_variance_of_beta_dist(settings: dict, var_beta_dist: Union[list, pt.Tensor], n_cases: int = 0) -> None:
    """
    plots the mean rewards received throughout the training periode and the corresponding standard deviation

    :param settings: dict containing all the paths etc.
    :param var_beta_dist: computed variance of the beta-function wrt episode
    :param n_cases: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    for c in range(n_cases):
        ax.plot(range(len(var_beta_dist[c])), var_beta_dist[c], color=settings["color"][c], label=settings["legend"][c])

    ax.set_ylabel("$mean$ $variance$ $of$ $beta-distribution$", usetex=True, fontsize=12)
    ax.set_xlabel("$episode$ $number$", usetex=True, fontsize=12)
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    plt.savefig("".join([settings["main_load_path"], setup["path_controlled"], "/plots/var_beta_distribution.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def merge_results_for_diff_seeds(data: list, n_seeds: int) -> dict:
    """
    merge the trajectories of the PPO-training episode-wise

    prerequisites: all trainings are done with the same setup (same number of workers etc. but e.g. trainings
    initialized with different seeds)

    :param data: the loaded training data from all cases which should be merged
    :param n_seeds: number of cases
    :return: a dictionary containing the merged data
    """
    n_traj = sum([data[seed]["n_workers"] for seed in range(n_seeds)])
    shape = (data[0]["cd"].size(0), data[0]["cd"].size(1), n_traj)
    cl, cd, rewards, actions, alpha, beta = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), \
                                            pt.zeros(shape), pt.zeros(shape)
    states = pt.zeros((data[0]["states"].size(0), data[0]["states"].size(1), data[0]["states"].size(2), n_traj))

    for seed in range(n_seeds):
        cd[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["cd"]
        cl[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["cl"]
        rewards[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["rewards"]
        alpha[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["alpha"]
        beta[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["beta"]
        actions[:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["actions"]
        states[:, :, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["states"]

    # sort back into dict
    merged_data = {"n_workers": n_traj, "network_data": [data[seed]["network_data"] for seed in range(len(data))],
                   "cl": cl, "cd": cd, "states": states, "actions": actions, "rewards": rewards, "alpha": alpha,
                   "beta": beta, "n_seeds": n_seeds}

    return merged_data


def plot_total_reward(settings: dict, reward_mean: list, reward_std: list, n_cases: int) -> None:
    """
    plot the total rewards of the complete training for each case

    :param settings: dict containing all the paths etc.
    :param reward_mean: mean total rewards received in the training
    :param reward_std: corresponding standard deviation of the total rewards received in the training
    :param n_cases: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for c in range(n_cases):
        ax.errorbar(c + 1, reward_mean[c], yerr=reward_std[c], color=settings["color"][c], fmt="o", capsize=5,
                    label=settings["legend"][c])

    ax.set_ylabel("$total$ $reward$", usetex=True, fontsize=12)
    ax.set_xlabel("$case$ $number$", usetex=True, fontsize=12)
    ax.set_xticks(range(1, n_cases + 1, 1))
    ax.legend(loc="lower right", framealpha=1.0, fontsize=10, ncol=2)
    plt.grid(which="major", axis="y", linestyle="--", alpha=0.85, color="black", lw=1)
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"], "/plots/total_rewards.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_numerical_setup(settings: dict) -> None:
    """
    plot the domain of the cylinder case

    :param settings: setup containing the import and save path
    :return: None
    """
    pattern = r"\d.\d+ \d.\d+ \d.\d+"
    path = "".join([settings["main_load_path"], settings["path_controlled"], settings["case_name"][0],
                    settings["path_final_results"]])
    with open("".join([path, settings["path_to_probes"], "p"]), "r") as f:
        loc = f.readlines()

    # get coordinates of probes, omit appending empty lists and map strings to floats
    coord = [re.findall(pattern, line) for line in loc if re.findall(pattern, line)]
    pos_probes = pt.tensor([list(map(float, i[0].split())) for i in coord])

    # get coordinates of domain and cylinder
    with open("".join([path, "system/blockMeshDict"]), "r") as f:
        loc = f.readlines()

    # structure in blockMeshDict always the same: [lengthX 2.2, lengthY 0.41, cylinderX 0.2, cylinderY 0.2, radius 0.05]
    l, h, pos_x, pos_y, r = [float(loc[i].strip(";\n").split()[1]) for i in range(16, 21)]

    # plot cylinder, probe locations and annotate the domain
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pos_probes[:, 0], pos_probes[:, 1], linestyle="None", marker="o", color="red", label="probes")
    # dummy point for legend
    ax.scatter(-10, -10, marker="o", color="black", alpha=0.4, label="cylinder")
    circle = Circle((pos_x, pos_y), radius=r, color="black", alpha=0.4)
    rectangle = Rectangle((0, 0), width=l, height=h, edgecolor="black", linewidth=2, facecolor="none")
    ax.add_patch(circle)
    ax.add_patch(rectangle)
    fig.legend(loc="lower right", framealpha=1.0, fontsize=10, ncol=2)
    ax.set_xlim(0, l)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])

    # annotate inlet & outlet
    plt.arrow(-0.05, -0.05, 0.1, 0.0, color="black", head_width=0.02, clip_on=False)
    plt.arrow(-0.05, -0.05, 0.0, 0.1, color="black", head_width=0.02, clip_on=False)
    plt.arrow(-0.1, h * 2 / 3 + 0.025, 0.075, -0.05, color="black", head_width=0.015, clip_on=False)
    plt.arrow(-0.1 + l, h * 2 / 3, 0.075, -0.05, color="black", head_width=0.015, clip_on=False)

    plt.annotate("$inlet$", (-0.15, h * 2 / 3 + 0.05), annotation_clip=False, usetex=True, fontsize=13)
    plt.annotate("$\\frac{x}{d}$", (0.1, -0.065), annotation_clip=False, usetex=True, fontsize=16)
    plt.annotate("$\\frac{y}{d}$", (-0.1, 0.065), annotation_clip=False, usetex=True, fontsize=16)
    plt.annotate("$outlet$", (-0.2 + l, h * 2 / 3 + 0.01), annotation_clip=False, usetex=True, fontsize=13)

    # annotate the dimensions & position of the domain
    pos = {"xy": [(0, h + 0.04), (0, h), (l, h), (pos_x - r - 0.01, pos_y - 0.1), (l, h), (l, 0), (l + 0.04, h),
                  (pos_x, pos_y + 0.9 * r), (0, pos_y)],
           "xytxt": [(l, h + 0.04), (0, h + 0.075), (l, h + 0.075), (pos_x + r + 0.01, pos_y - 0.1), (l + 0.075, h),
                     (l + 0.075, 0),
                     (l + 0.04, 0), (pos_x, h), (pos_x - 0.9 * r, pos_y)],
           "style": [("<->", "-"), ("-", "--"), ("-", "--"), ("<->", "-"), ("-", "--"), ("-", "--"), ("<->", "-"),
                     ("<->", "-"), ("<->", "-")]
           }
    for i in range(len(pos["style"])):
        plt.annotate("", xy=pos["xy"][i], xytext=pos["xytxt"][i],
                     arrowprops=dict(arrowstyle=pos["style"][i][0], color="black", linestyle=pos["style"][i][1]),
                     annotation_clip=False)

    plt.annotate(f"${l / (2 * r)}$", (l / 2, h + 0.07), annotation_clip=False, usetex=True, fontsize=12)
    plt.annotate(f"${h / (2 * r)}$", (l + 0.07, h / 2), annotation_clip=False, usetex=True, fontsize=12)
    plt.annotate("$d$", (pos_x - r / 4, pos_y - 3 * r), usetex=True, fontsize=12)
    plt.annotate("${:.2f}$".format((h - (pos_y + r)) / (2 * r)), (pos_x + 0.025, pos_y + 2.25 * r), usetex=True,
                 fontsize=12)
    plt.annotate("${:.2f}$".format((pos_x - r) / (2 * r)), (pos_x - 3.25 * r, pos_y + 0.5 * r), usetex=True,
                 fontsize=12)

    ax.plot((pos_x - r, pos_x - r), (pos_y, pos_y - 0.15), color="black", linestyle="--", lw=1)
    ax.plot((pos_x + r, pos_x + r), (pos_y, pos_y - 0.15), color="black", linestyle="--", lw=1)
    ax.plot(pos_x, pos_y, marker="+", color="black")

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.92)
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"], "/plots/domain_setup.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/",     # top-level directory containing all the cases
        "path_to_probes": r"postProcessing/probes/0/",              # path to the file containing trajectories of probes
        "path_uncontrolled": r"robust_MB_DRL_for_flow_control/run/cylinder2D_uncontrolled/"
                             r"cylinder2D_uncontrolled_Re100/",             # path to reference case
        "path_controlled": r"drlfoam/examples/test_MF_vs_MB_DRL/",          # main path to all the controlled cases
        "path_final_results": r"results_best_policy/",                      # path to the results using the best policy
        "case_name": ["MF_DRL_buffer8_2sec/", "MB_DRL_buffer8_2sec/", "MB_DRL_buffer8_2sec_2nd/"],
        "avg_over_cases": False,                                # if cases should be averaged over, e.g. different seeds
        "plot_final_res": False,                        # if the final policy already ran in openfoam, plot the results
        "param_study": False,           # flag if parameter study, only used for generating legend entries automatically
        "color": ["blue", "red", "green", "darkviolet"],        # colors for the cases, uncontrolled = black
        "legend": ["MF-DRL ($b = 8$, $l = 2s$)", "MB-DRL ($b = 8$, $l = 2s$, $N_{t, input} = 30$)",
                   "MB-DRL ($b = 8$, $l = 2s$, $N_{t, input} = 15$)"]
    }

    # create directory for plots
    if not os.path.exists(setup["main_load_path"] + setup["path_controlled"] + "plots"):
        os.mkdir(setup["main_load_path"] + setup["path_controlled"] + "plots")

    # load all the data
    all_data = load_all_data(setup)

    # average the trajectories episode-wise
    averaged_data = average_results_for_each_case(all_data)

    # for parameter study: generate legend entries automatically
    if setup["param_study"]:
        if setup["avg_over_cases"]:
            setup["legend"] = [f"b = {averaged_data['buffer_size'][c]}, l = {averaged_data['len_traj'][c]} s" for c in
                               range(len(averaged_data["len_traj"]))]
        else:
            setup["legend"] = [f"seed = {c}" for c in range(len(averaged_data["len_traj"]))]

    # plot variance of the beta-distribution wrt episodes
    plot_variance_of_beta_dist(setup, averaged_data["var_beta_fct"], n_cases=len(setup["case_name"]))

    # plot mean rewards wrt to episode
    plot_rewards_vs_episode(setup, reward_mean=averaged_data["mean_rewards"], reward_std=averaged_data["std_rewards"],
                            n_cases=len(setup["case_name"]))

    # plot mean cl and cd wrt to episode
    plot_results_vs_episode(setup, cd_mean=averaged_data["mean_cd"], cd_std=averaged_data["std_cd"],
                            cl_mean=averaged_data["mean_cl"], cl_std=averaged_data["std_cl"],
                            actions_mean=averaged_data["mean_actions"], actions_std=averaged_data["std_actions"],
                            n_cases=len(setup["case_name"]), plot_action=False)

    # plot total rewards received in the training
    plot_total_reward(setup, averaged_data["tot_mean_rewards"], averaged_data["tot_std_rewards"],
                      n_cases=len(setup["case_name"]))

    # do frequency analysis of the cd- and cl-trajectories wrt episode number for each case
    for case in range(len(setup["case_name"])):
        analyze_frequencies_ppo_training(setup, all_data[case], case=case + 1)

    # if the cases are run in openfoam using the trained network (using the best policy), plot the results
    if setup["plot_final_res"]:
        # plot the numerical setup for one case, assuming it's the same for all cases
        plot_numerical_setup(setup)

        # import the trajectory of the uncontrolled case
        uncontrolled = pd.read_csv(setup["main_load_path"] + setup["path_uncontrolled"] +
                                   r"postProcessing/forces/0/coefficient.dat", skiprows=13, header=0, sep=r"\s+",
                                   usecols=[0, 1, 3], names=["t", "cd", "cl"])

        controlled, traj = [], []
        for case in range(len(setup["case_name"])):
            # import the trajectories of the controlled cases
            controlled.append(pd.read_csv(setup["main_load_path"] + setup["path_controlled"] + setup["case_name"][case]
                                          + setup["path_final_results"] + r"postProcessing/forces/0/coefficient.dat",
                                          skiprows=13, header=0, sep=r"\s+", usecols=[0, 1, 2],
                                          names=["t", "cd", "cl"]))
            traj.append(pd.read_csv(setup["main_load_path"] + setup["path_controlled"] + setup["case_name"][case] +
                                    setup["path_final_results"] + r"/trajectory.csv", header=0, sep=r",",
                                    usecols=[0, 1, 2, 3], names=["t", "omega", "alpha", "beta"]))

        # plot cl and cd of the controlled cases vs. the uncontrolled cylinder flow
        plot_cl_cd_alpha_beta(setup, controlled, uncontrolled, plot_coeffs=True)

        # plot omega of the controlled cases
        plot_omega(setup, traj)

        # plot alpha and beta of the controlled cases
        plot_cl_cd_alpha_beta(setup, traj, plot_coeffs=False)

        # analyze frequency spectrum of cl- and cd-trajectories
        analyze_frequencies_final_result(setup, uncontrolled, controlled)
