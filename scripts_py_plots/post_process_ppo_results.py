"""
    brief:
        - post-processes and plot the result of the training using PPO and CFD as an environment
        - plots the results of the controlled case using the best policy in comparison to the uncontrolled case

    dependencies:
        - None

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to conduct a training
          (https://github.com/OFDataCommittee/drlfoam)
        - execution of simulation for the best policy from training, also results of a simulation without control
"""
import os
import pickle
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from typing import Union


def load_trajectory_data(path: str) -> dict:
    """
    :brief: load observations_*.pkl files containing all the data generated during training and sort them into a dict
    :param path: path to directory containing the files
    :return: dict with actions, states, cl, cd. Each parameter contains one tensor with the length of N_episodes, each
             entry has all the trajectories sampled in this episode (cols = N_trajectories, rows = length_trajectories)
    """
    # for training an environment model it doesn't matter in which order files are read in -> no sorting required
    files = glob(path + "observations_*.pkl")
    observations = [pickle.load(open(file, "rb")) for file in files]
    traj_length = len(observations[0][0]["actions"])

    data = {"n_workers": len(observations[0])}

    # sort the trajectories from all workers wrt the episode
    shape = (traj_length, data["n_workers"])
    actions, cl, cd, rewards, alpha, beta = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), \
                                            pt.zeros(shape), pt.zeros(shape)
    states = pt.zeros((shape[0], observations[0][0]["states"].size()[1], shape[1]))
    shape = (len(observations), traj_length, data["n_workers"])
    data["actions"], data["cl"], data["cd"], = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
    data["rewards"], data["alpha"], data["beta"], = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
    data["states"] = pt.zeros((shape[0], shape[1], observations[0][0]["states"].size()[1], shape[2]))
    for episode in range(len(observations)):
        for worker in range(len(observations[episode])):
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

    # load value-, policy and MSE losses of PPO training
    data["network_data"] = pickle.load(open(path + "training_history.pkl", "rb"))

    return data


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
                "std_beta": []}
    for case in range(len(data)):
        n_episodes, len_trajectory = data[case]["actions"].size()[0], data[case]["actions"].size()[1]

        # calculate avg. and std. dev. of all trajectories within episode
        # TO_DO: find more efficient way of sorting the parameters into dict
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

    for case in range(n_cases):
        for i in range(n_subfig):
            if i == 0:
                ax[i].plot(range(len(cl_mean[case])), cl_mean[case], color=settings["color"][case],
                           label=settings["legend"][case])
                ax[i].fill_between(range(len(cl_mean[case])), cl_mean[case] - cl_std[case],
                                   cl_mean[case] + cl_std[case],
                                   color=settings["color"][case], alpha=0.3)
                ax[i].set_ylabel("$mean$ $lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)

            elif i == 1:
                ax[i].plot(range(len(cd_mean[case])), cd_mean[case], color=settings["color"][case])
                ax[i].fill_between(range(len(cd_mean[case])), cd_mean[case] - cd_std[case],
                                   cd_mean[case] + cd_std[case],
                                   color=settings["color"][case], alpha=0.3)
                ax[i].set_ylabel("$mean$ $drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)

            elif plot_action:
                ax[i].plot(range(len(actions_mean[case])), actions_mean[case], color=settings["color"][case])
                ax[i].fill_between(range(len(actions_mean[case])), actions_mean[case] - actions_std[case],
                                   actions_mean[case] + actions_std[case], color=settings["color"][case], alpha=0.3)
                ax[i].set_ylabel("$mean$ $action$ $\qquad \omega$", usetex=True, fontsize=13)

            ax[i].set_xlabel("$episode$ $number$", usetex=True, fontsize=13)

    fig.suptitle(" ", usetex=True, fontsize=14)
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    fig.subplots_adjust(wspace=0.25)
    plt.savefig(settings["main_load_path"] + setup["path_controlled"] + "/plots/coefficients_vs_episode.png", dpi=600)
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
    for case in range(n_cases):
        ax.plot(range(len(reward_mean[case])), reward_mean[case], color=settings["color"][case],
                label=settings["legend"][case])
        ax.fill_between(range(len(reward_mean[case])), reward_mean[case] - reward_std[case],
                        reward_mean[case] + reward_std[case],
                        color=settings["color"][case], alpha=0.3)

    ax.set_ylabel("$mean$ $reward$", usetex=True, fontsize=12)
    ax.set_xlabel("$episode$ $number$", usetex=True, fontsize=12)
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    plt.savefig(settings["main_load_path"] + setup["path_controlled"] + "/plots/rewards_vs_episode.png", dpi=600)
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

    for case in n_cases:
        for i in range(2):
            if i == 0:
                if case == 0:
                    ax[i].plot(uncontrolled_case[keys[0]], uncontrolled_case[keys[1]], color="black",
                               label="uncontrolled")
                else:
                    ax[i].plot(controlled_cases[case - 1][keys[0]], controlled_cases[case - 1][keys[1]],
                               color=setup["color"][case - 1], label=settings["legend"][case - 1])
                ax[i].set_ylabel(ylabels[0], usetex=True, fontsize=13)
            else:
                if case == 0:
                    ax[i].plot(uncontrolled_case[keys[0]], uncontrolled_case[keys[2]], color="black")
                else:
                    ax[i].plot(controlled_cases[case - 1][keys[0]], controlled_cases[case - 1][keys[2]],
                               color=settings["color"][case - 1])
                ax[i].set_ylabel(ylabels[1], usetex=True, fontsize=13)

            ax[i].set_xlabel("$time$ $[s]$", usetex=True, fontsize=13)
    fig.suptitle("", usetex=True, fontsize=14)
    fig.tight_layout(pad=1.5)
    fig.legend(loc="upper right", framealpha=1.0, fontsize=11, ncol=len(settings["case_name"]) + 1)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(settings["main_load_path"] + f"/plots/{save_name}.png", dpi=600)
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
    for case in range(len(settings["case_name"])):
        ax.plot(controlled_cases[case]["t"], controlled_cases[case]["omega"], color=settings["color"][case],
                label=settings["legend"][case])

    ax.set_ylabel("$action$ $\omega$", usetex=True, fontsize=13)
    ax.set_xlabel("$time$ $[s]$", usetex=True, fontsize=13)
    fig.suptitle("", usetex=True, fontsize=14)
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=11, ncol=len(settings["case_name"]))
    plt.savefig(settings["main_load_path"] + f"/plots/omega_controlled_cases.png", dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/",     # top-level directory containing all the cases
        "path_to_probes": r"postProcessing/probes/0/",              # path to the file containing trajectories of probes
        "path_uncontrolled": r"run/cylinder2D_uncontrolled/cylinder2D_uncontrolled_Re100/",     # path to reference case
        "path_controlled": r"drlfoam/examples/",                    # main path to all the controlled cases
        "path_final_results": r"results_best_policy/",              # path to the results using the best policy
        "case_name": ["test_training2/", "test_training3/"],        # dirs containing the training results
        "avg_over_cases": False,                                # if cases should be averaged over, e.g. different seeds
        "plot_final_res": True,                          # if the final policy already ran in openfoam, plot the results
        "color": ["blue", "red", "green", "darkviolet"],         # colors for the different cases, uncontrolled = black
        "legend": ["test 2", "test 3"]                            # legend entries
    }
    # create directory for plots
    if not os.path.exists(setup["main_load_path"] + "/plots"):
        os.mkdir(setup["main_load_path"] + "/plots")

    # load the results of the training
    loaded_data, controlled, traj = [], [], []
    for case in range(len(setup["case_name"])):
        loaded_data.append(load_trajectory_data(setup["main_load_path"] + setup["path_controlled"] +
                                                setup["case_name"][case]))

    if not setup["avg_over_cases"]:
        # average the trajectories episode-wise
        averaged_data = average_results_for_each_case(loaded_data)

        # plot mean rewards wrt to episode
        plot_rewards_vs_episode(setup, reward_mean=averaged_data["mean_rewards"],
                                reward_std=averaged_data["std_rewards"], n_cases=len(setup["case_name"]))

        # plot mean cl and cd wrt to episode
        plot_results_vs_episode(setup, cd_mean=averaged_data["mean_cd"], cd_std=averaged_data["std_cd"],
                                cl_mean=averaged_data["mean_cl"], cl_std=averaged_data["std_cl"],
                                actions_mean=averaged_data["mean_actions"], actions_std=averaged_data["std_actions"],
                                n_cases=len(setup["case_name"]), plot_action=False)

    else:
        # TO_DO: average over different seed values per case
        pass

    # TO_DO: import and plot training & validation losses of the value- and policy-networks

    # if the cases are run in openfoam using the trained network (using the best policy), plot the results
    if setup["plot_final_res"]:
        # import the trajectory of the uncontrolled case
        uncontrolled = pd.read_csv(setup["main_load_path"] + setup["path_uncontrolled"] +
                                   r"postProcessing/forces/0/coefficient.dat", skiprows=13, header=0, sep=r"\s+",
                                   usecols=[0, 1, 3], names=["t", "cd", "cl"])

        for case in range(len(setup["case_name"])):
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
