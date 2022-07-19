"""
 this script:
  - computes mean rewards and std. deviation during training
  - computes mean cl, cd (or other coefficients) and the corresponding std. deviation during training
  - plot rewards vs. epochs for arbitrary number of cases
  - plot cl, cd vs. epochs for arbitrary number of cases
"""
import os
import glob
import pandas as pd
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt


def import_coefficients(dirs: str, epochs: int):
    """
    :brief: imports cl-, cd- coefficients for all trajectories and epochs, computes mean and std. deviation of it
    :param dirs: case which was trained
    :param epochs: total number of epochs run for the training of this case
    :return: mean coefficients and their std. deviation for all epochs
    """

    mean_coeffs, std_coeffs = np.zeros((epochs, 2)), np.zeros((epochs, 2))
    sampleDirs = natsorted(glob.glob(dirs + r"/Data/sample_*/"))

    # get cl and cd values for all trajectories in each sample-directory (sample-dir = number of epoch)
    for i in range(len(sampleDirs)):
        # temp. list for cl, cd -> saves data temporarily from all trajectories, is cleared every new epoch,
        # number of trajectory may differ within epochs due to failed trajectories
        trajectory_dirs, cl, cd = sorted(glob.glob(sampleDirs[i] + r"trajectory_*/")), [], []

        for j in range(len(trajectory_dirs)):
            data = pd.read_csv(trajectory_dirs[j] + "coefficient.dat", skiprows=13, header=0, sep="\s+", usecols=[1, 3],
                               names=["cd", "cl"])
            cl.append(data["cl"])
            cd.append(data["cd"])

        # mean over all trajectories per epoch and the corresponding std. deviation
        mean_coeffs[i] = np.mean(np.concatenate(cl, axis=0), axis=0), np.mean(np.concatenate(cd, axis=0), axis=0)
        std_coeffs[i] = np.std(np.concatenate(cl, axis=0), axis=0), np.std(np.concatenate(cd, axis=0), axis=0)

    # return matrix with rows containing the epochs and columns [mean_cl, mean_cd, std_cl, std_cd]
    return np.concatenate((mean_coeffs, std_coeffs), axis=1)


if __name__ == "__main__":
    # Setup
    setup = {
        "path": r"/home/janis/ml-cfd-lecture/exercises/UE11/influenceManualSeed/",
        "title": "UE 11: influence seed value",
        "colors": ["blue", "green", "red", "darkviolet", "magenta"],
        "legend_entries": ["seed = 0", "seed = 10", "seed = 25"]
    }

    if not os.path.exists(setup["path"] + "plots/"):
        os.mkdir(setup["path"] + "plots/")

    # import rewards of all cases, natsort() bc otherwise ist sorted like  [... 19.npy, 2.npy, 20.npy, ...]
    folders = sorted(glob.glob(setup["path"] + r"drl_control_cylinder*"))
    files = [natsorted(glob.glob(i + r"/results/evaluation/evaluations_*.npy")) for i in folders]
    alloc_size = (len(folders), max([len(i) for i in files]))
    meanReward, stdDeviation, len_epoch = np.zeros(alloc_size), np.zeros(alloc_size), []

    for i in range(len(folders)):
        for j in range(len(files[i])):
            # reward array is saved as: [mean_reward_complete_array, std_reward_complete_array,
            #                            mean_reward_last_10_values, std_reward_last_10_values,
            #                            mean_reward_last_100_values, std_reward_last_100_values]
            meanReward[i][j] = np.load(files[i][j])[0]
            stdDeviation[i][j] = np.load(files[i][j])[1]

        # save number of epochs bc epochs may differ for each case and matrix is allocated for max. number of epochs
        len_epoch.append(j + 1)

    # plot mean rewards and their std. deviation
    fig1 = plt.figure(num=1, figsize=(10, 6))
    for i in range(len(folders)):
        plt.plot(range(len_epoch[i]), meanReward[i][:len_epoch[i]], color=setup["colors"][i],
                 label=setup["legend_entries"][i])
        plt.fill_between(range(len_epoch[i]), meanReward[i][:len_epoch[i]] - stdDeviation[i][:len_epoch[i]],
                         meanReward[i][:len_epoch[i]] + stdDeviation[i][:len_epoch[i]], color=setup["colors"][i], alpha=0.3)
    plt.xlabel("PPO iteration")
    plt.ylabel("mean reward")
    plt.title(setup["title"])
    plt.legend(loc="lower right", framealpha=1.0, fontsize=10)
    plt.savefig(setup["path"] + f"plots/meanReward.png", dpi=600)

    # import and plot mean cl and cd with their corresponding std. deviations for each case
    cd_cl_coeffs = []
    for i in range(len(folders)):
        cd_cl_coeffs.append(import_coefficients(folders[i], len_epoch[i]))

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(len(cd_cl_coeffs)):
        # mean cl and std. deviation
        axes2[0].plot(range(len_epoch[i]), cd_cl_coeffs[i][:, 0], color=setup["colors"][i], label=setup["legend_entries"][i])
        # axes2[0].fill_between(range(len_epoch[i]), cd_cl_coeffs[i][:, 0] - cd_cl_coeffs[i][:, 2], cd_cl_coeffs[i][:, 0]
        #                       + cd_cl_coeffs[i][:, 2], color=setup["colors"][i], alpha=0.3)
        # mean cd and std. deviation
        axes2[1].plot(range(len_epoch[i]), cd_cl_coeffs[i][:, 1], color=setup["colors"][i])
        axes2[1].fill_between(range(len_epoch[i]), cd_cl_coeffs[i][:, 1] - cd_cl_coeffs[i][:, 3], cd_cl_coeffs[i][:, 1]
                              + cd_cl_coeffs[i][:, 3], color=setup["colors"][i], alpha=0.3)

    axes2[0].set_xlabel("PPO iteration")
    axes2[1].set_xlabel("PPO iteration")
    # axes2[0].set_ylabel("$\\bar{c}_l$, $\sigma_{c_l}$")
    axes2[0].set_ylabel("mean lift coefficient \t$\\bar{c}_l$")
    axes2[1].set_ylabel("drag coefficient \t$\\bar{c}_d$, $\sigma_{c_d}$")
    fig2.suptitle(setup["title"])
    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0.25)
    fig2.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=3, )
    plt.savefig(setup["path"] + f"plots/cl_cd.png", dpi=600)
    plt.show()
