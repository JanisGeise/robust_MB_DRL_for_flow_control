"""
    brief:
        - plots the influence of the ratio of MF episodes and MB episodes used throughout the PPO-training routine
        - computes the optimal ratio between number of MF episodes and number of MB episodes wrt achieved rewards on the
          one hand and run time on the other hand
          -> goal is to minimize the run times and maximize the rewards (or minimize 1/rewards). Global minimum can
             then be found by adding the curves of 1/rewards and run time

    dependencies:
        - 'plot_ppo_results.py', located in the 'scripts_py_plots' directory
        - 'utils.py', located in the 'test_env_models' directory

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""

from test_env_models.utils import normalize_data
from scripts_py_plots.plot_ppo_results import *
from influence_buffer_and_trajectory_length import get_mean_run_time


def plot_influence_ratio(settings: dict, runtime: pt.Tensor, reward: pt.Tensor, ratios: list) -> None:
    """
    plot the runtime and inverted rewards wrt the ratio of episodes ran with environment model to episodes ran
    model-free in CFD, also plot sum to determine optimum wrt to rewards and runtime

    :param settings: dict containing all the paths etc.
    :param runtime: run times of the cases normalized to [0, 1]
    :param reward: mean inverted(!) rewards normalized to [0, 1]
    :param ratios: number of cases to compare (= number of imported data)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.plot(ratios, reward, color="green", label="inverted rewards $\left[\\frac{1}{r^*}\\right]$")
    ax.plot(ratios, runtime, color="blue", label="run time $\left[t^*\\right]$")
    ax.plot(ratios, runtime + reward, color="black", label="$\sum\,\left(\\frac{1}{r^*}, t^*\\right)$")
    ax.plot(ratios[pt.argmin(runtime + reward)], pt.min(runtime + reward), color="red", linestyle="none", marker="o",
            label="$min \left[\sum\,\left(\\frac{1}{r^*}, t^*\\right)\\right]$")
    ax.set_ylabel("$mean$ $quantities$", usetex=True, fontsize=12)
    ax.set_xlabel("$\\frac{N_{episodes}(MB)}{N_{episodes}(MF)}$", usetex=True, fontsize=12)
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    ax.set_title("$t^*$, $r^*$ $\in$ $[0, 1]$")
    fig.tight_layout()
    plt.savefig("".join([settings["main_load_path"], settings["path_controlled"], "/plots/MB_vs_MF.png"]),
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def data_loader(settings: dict) -> dict:
    """
    loads all data, normalizes rewards, runtimes , cd and cl and to [0, 1]

    :param settings: setup containing all paths
    :return: dict with normalized run times, cl, cd and INVERTED(!) rewards
    """
    # load all the data and average the trajectories episode-wise
    avg_data = average_results_for_each_case(load_all_data(setup))

    """
    # get the run times (when cases run on cluster, for now do it manually)
    TODO: if avg. over seeds then consider mean and std for each case, also in plot!
    runtimes = [get_mean_run_time(("".join([settings["runtimes"], settings["case_name"][c]]))) for c in 
                settings["case_name"]]
    """

    # scale everything to [0, 1]
    rt = normalize_data(pt.tensor(settings["runtimes"]))[0]
    cd = normalize_data(pt.tensor(avg_data["tot_mean_cd"]))[0]
    cl = normalize_data(pt.tensor(avg_data["tot_mean_cl"]))[0]
    inverse_r = normalize_data(1 / pt.tensor(avg_data["tot_mean_rewards"]))[0]

    return {"mean_cl": cl, "mean_cd": cd, "mean_rewards": inverse_r, "runtimes": rt,
            "n_cases": len(settings["case_name"]), "ratio_MB_MF_episodes": avg_data["ratio_MB_MF"]}


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/",  # top-level directory containing all the cases
        "path_controlled": r"drlfoam/examples/test_MF_vs_MB_DRL/one_cfd_episode_for_training/influence_ratio_MF_MB/",
        "case_name": ["all_MF/", "MB_MF_4/", "MB_MF_5/", "MB_MF_6/", "MB_MF_10/"],
        "avg_over_cases": False,                                # if cases should be averaged over, e.g. different seeds
        "env": "local",                      # 'local' or 'cluster', if local: specify runtimes, otherwise logs are used
        "runtimes": [25817, 15213, 11520, 7483, 5701]          # run times [s]
    }

    # create directory for plots
    if not path.exists(setup["main_load_path"] + setup["path_controlled"] + "plots"):
        mkdir(setup["main_load_path"] + setup["path_controlled"] + "plots")

    # load the data
    data = data_loader(setup)

    # plot mean rewards wrt to episode
    plot_influence_ratio(setup, runtime=data["runtimes"], reward=data["mean_rewards"],
                         ratios=setup["ratio_MB_MF_episodes"])
