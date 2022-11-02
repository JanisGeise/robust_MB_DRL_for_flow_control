"""
    brief:
        - post-processes and plots the results of the training using PPO, either for MF-DRL or MB-DRL
        - plots the results of the controlled case using the best policy in comparison to the uncontrolled case

    dependencies:
        - 'ppo_data_loader.py' for handling the loading, sorting and merging of all training data
        - 'analyze_frequency_spectrum.py' for plotting the frequency spectrum of cl- and cd of PPO-training and the
           final results

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)

    optional:
        - execution of simulation for the best policy from training, also results of a simulation without control
"""
import re
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union
from os import mkdir, path
from matplotlib.patches import Circle, Rectangle

from ppo_data_loader import *
from analyze_frequency_spectrum import analyze_frequencies_final_result, analyze_frequencies_ppo_training,\
    analyze_frequencies_probes_final_result


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
                               color=settings["color"][c - 1], label=settings["legend"][c - 1])
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
    fig.subplots_adjust(top=0.91)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=1)
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
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
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
    with open("".join([path, settings["path_to_probes"]]), "r") as f:
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


def plot_train_validation_loss(settings: dict, mse_train: Union[list, pt.Tensor], mse_val: Union[list, pt.Tensor],
                               mse_train_cd: Union[list, pt.Tensor], mse_val_cd: Union[list, pt.Tensor],
                               std_dev_train: Union[list, pt.Tensor], std_dev_val: Union[list, pt.Tensor],
                               std_dev_train_cd: Union[list, pt.Tensor], std_dev_val_cd: Union[list, pt.Tensor],
                               case: int = 1) -> None:
    """
    plots the avg. train- and validation loss and the corresponding std. deviation of the environment models wrt to
    epochs

    :param settings: path where the plot should be saved
    :param mse_train: tensor containing the (mean) training loss of the cl-p env. model
    :param mse_val: tensor containing the (mean) validation loss of the cl-p env. model
    :param mse_train_cd: tensor containing the (mean) training loss of the cd env. model
    :param mse_val_cd: tensor containing the (mean) validation loss of the cd env. model
    :param std_dev_train: tensor containing the (std. deviation) training loss of the cl-p env. model
    :param std_dev_val: tensor containing the (std. deviation) validation loss of the cl-p env. model
    :param std_dev_train_cd: tensor containing the (std. deviation) training loss of the cd env. model
    :param std_dev_val_cd: tensor containing the (std. deviation) validation loss of the cd env. model
    :param case: name to append for savin img
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(2):
        if i == 0:
            x = range(len(mse_train))
            ax[i].plot(x, mse_train, color="blue")
            ax[i].plot(x, mse_val, color="red")
            ax[i].fill_between(x, mse_val - std_dev_val, mse_val + std_dev_val, color="red", alpha=0.3)
            ax[i].fill_between(x, mse_train - std_dev_train, mse_train + std_dev_train, color="blue", alpha=0.3)
            ax[i].set_ylabel("$MSE$ $loss$", usetex=True, fontsize=13)
            ax[i].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
            ax[i].set_title("$environment$ $model$ $for$ $c_l$ $\&$ $p_i$", usetex=True, fontsize=14)
            ax[i].set_yscale("log")

        else:
            x = range(len(mse_train_cd))
            ax[i].plot(x, mse_train_cd, color="blue", label="training loss")
            ax[i].plot(x, mse_val_cd, color="red", label="validation loss")
            ax[i].fill_between(x, mse_val_cd - std_dev_val_cd, mse_val_cd + std_dev_val_cd, color="red", alpha=0.3)
            ax[i].fill_between(x, mse_train_cd - std_dev_train_cd, mse_train_cd + std_dev_train_cd, color="blue",
                               alpha=0.3)
            ax[i].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
            ax[i].set_title("$environment$ $model$ $for$ $c_d$", usetex=True, fontsize=14)
            ax[i].set_yscale("log")
            ax[i].set_ylabel("$MSE$ $loss$", usetex=True, fontsize=13)

    ax[1].legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"],
                         f"/plots/train_val_losses_case{case}.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_trajectories(settings: dict, data: list[dict], number: int, e: int = 1) -> None:
    """
    plots the trajectory of cl and cd for different episodes of the training, meant to use for either comparing MF-
    trajectories to trajectories generated by the environment models or comparing trajectories from environment models
    run with different settings to each other

    :param settings: setup containing all the paths etc.
    :param data: trajectory data to plot
    :param number: number of the trajectory within the data set (either within the episode or in total)
    :param e: episode number
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    epochs = range(len(data[0]["cl"][1, :, number]))
    for n in range(len(data)):
        for i in range(3):
            try:
                if i == 0:
                    # 2nd episode is always MB (if MB-DRL was used)
                    ax[i].plot(epochs, data[n]["cl"][e, :, number], color=settings["color"][n],
                               label=f"{settings['legend'][n]}, episode {e + 1}")
                    ax[i].set_ylabel("$lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)
                elif i == 1:
                    ax[i].plot(epochs, data[n]["cd"][e, :, number], color=settings["color"][n])
                    ax[i].set_ylabel("$drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)
                else:
                    ax[i].plot(epochs, data[n]["actions"][e, :, number], color=settings["color"][n])
                    ax[i].set_ylabel("$action$ $\qquad \omega$", usetex=True, fontsize=13)
                ax[i].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
            except IndexError:
                print("omit plotting trajectories of failed cases")
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    fig.subplots_adjust(wspace=0.25, top=0.90)
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"],
                         f"/plots/comparison_traj_cl_cd_{e}.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"../drlfoam/",                           # top-level directory
        "path_to_probes": r"postProcessing/probes/0/p",              # path to probes (usually always the same)
        "path_uncontrolled": r"run/uncontrolled/",                  # path to uncontrolled reference case
        "path_controlled": r"examples/run_training/",               # path to controlled cases
        "path_final_results": r"results_best_policy/",              # path to the results using the best policy
        # "case_name": ["seed0/", "seed1/", "seed2/"],
        "case_name": ["e80_r10_b10_f8_MF/", "e80_r10_b10_f5_MB/"],  # dirs with the training results (controlled)
        "e_trajectory": [0, 1, 4, 75, 79],              # for which episodes should a trajectory be plotted (cl & cd)
        "n_probes": 12,         # number of probes placed in flow field
        "avg_over_cases": True,                                 # if cases should be averaged over, e.g. different seeds
        "plot_final_res": False,                        # flag for plotting the results using final policy, if available
        "param_study": True,           # flag if parameter study, only used for generating legend entries automatically
        "color": ["blue", "red", "green", "darkviolet"],  # line colors, uncontrolled = black (if plot final res = True)
        "legend": ["MF-DRL #1", "MB-DRL #1"]               # legend entries, uncontrolled is added automatically if True
    }

    # create directory for plots
    if not path.exists(setup["main_load_path"] + setup["path_controlled"] + "plots"):
        mkdir(setup["main_load_path"] + setup["path_controlled"] + "plots")

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

    # compare trajectories over the course of the training
    for e in setup["e_trajectory"]:
        plot_cl_cd_trajectories(setup, all_data, number=1, e=e)

    # do frequency analysis of the cd- and cl-trajectories wrt episode number for each case
    for case in range(len(setup["case_name"])):
        analyze_frequencies_ppo_training(setup, all_data[case], case=case + 1)

        # also plot training- and validation losses of the environment models, if MB-DRL was used
        if "train_loss_cd" in all_data[case]:
            keys = 2 * ["train_loss_cl_p", "val_loss_cl_p", "train_loss_cd", "val_loss_cd"]
            key = [f"mean_" + k if idx < 4 else f"std_" + k for idx, k in enumerate(keys)]
            plot_train_validation_loss(setup, averaged_data["losses"][case][key[0]],
                                       averaged_data["losses"][case][key[1]], averaged_data["losses"][case][key[2]],
                                       averaged_data["losses"][case][key[3]], averaged_data["losses"][case][key[4]],
                                       averaged_data["losses"][case][key[5]], averaged_data["losses"][case][key[6]],
                                       averaged_data["losses"][case][key[7]], case + 1)

    # if the cases are run in openfoam using the trained network (using the best policy), plot the results
    if setup["plot_final_res"]:
        # plot the numerical setup for one case, assuming it's the same for all cases
        plot_numerical_setup(setup)

        # import the trajectory of the uncontrolled case
        uncontrolled = pd.read_csv("".join([setup["main_load_path"], setup["path_uncontrolled"],
                                           r"postProcessing/forces/0/coefficient.dat"]), skiprows=13, header=0,
                                   sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"])

        controlled, traj = [], []
        for case in range(len(setup["case_name"])):
            # import the trajectories of the controlled cases
            controlled.append(pd.read_csv("".join([setup["main_load_path"], setup["path_controlled"],
                                                   setup["case_name"][case], setup["path_final_results"],
                                                   r"postProcessing/forces/0/coefficient.dat"]), skiprows=13, header=0,
                                          sep=r"\s+", usecols=[0, 1, 2], names=["t", "cd", "cl"]))
            traj.append(pd.read_csv("".join([setup["main_load_path"], setup["path_controlled"],
                                             setup["case_name"][case], setup["path_final_results"],
                                             r"/trajectory.csv"]), header=0, sep=r",", usecols=[0, 1, 2, 3],
                                    names=["t", "omega", "alpha", "beta"]))

        # plot cl and cd of the controlled cases vs. the uncontrolled cylinder flow
        plot_cl_cd_alpha_beta(setup, controlled, uncontrolled, plot_coeffs=True)

        # plot omega of the controlled cases
        plot_omega(setup, traj)

        # plot alpha and beta of the controlled cases
        plot_cl_cd_alpha_beta(setup, traj, plot_coeffs=False)

        # analyze frequency spectrum of cl- and cd-trajectories, therefore insert empty list int traj. data, so the idx
        # matches with the other data (since uncontrolled case hass no alpha, beta, omega)
        traj.insert(0, [])
        analyze_frequencies_final_result(setup, uncontrolled, controlled, traj)

        # analyze frequency spectrum of probes
        analyze_frequencies_probes_final_result(setup, p_uncontrolled, p_controlled, n_probes=setup["n_probes"])
