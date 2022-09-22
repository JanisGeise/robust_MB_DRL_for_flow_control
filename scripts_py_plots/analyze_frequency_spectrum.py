"""
    brief:
        - analyze frequencies within the cl- and cd-trajectories of the sampled data for each episode
        - plots the results wrt the episode

    dependencies:
        - None

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to conduct a training
          (https://github.com/OFDataCommittee/drlfoam)
        - execution of simulation for the best policy from training, also results of a simulation without control
"""
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union
from scipy.signal import welch


def analyze_frequencies_ppo_training(settings: dict, data: dict, sampling_freq: int = 100, case: int = 0) -> None:
    """
    analyzes the frequency spectrum of the sampled trajectories wrt the episode

    :param settings: setup containing all the paths etc.
    :param data: the data containing the loaded trajectories of the PPO training
    :param sampling_freq: number of sampled data points per second (epochs per second CFD time)
    :param case: number of the imported case, corresponds to legend entry
    :return: None
    """
    len_traj = data["cd"].size()[1]
    freq_cd = pt.zeros((data["cd"].size()[0], data["n_workers"], int(len_traj * 0.5) + 1))
    amp_cd = pt.zeros((int(len_traj * 0.5) + 1, data["cd"].size()[0], data["n_workers"]))
    freq_cl = pt.zeros((data["cl"].size()[0], data["n_workers"], int(len_traj * 0.5) + 1))
    amp_cl = pt.zeros((int(len_traj * 0.5) + 1, data["cl"].size()[0], data["n_workers"]))

    for episode in range(data["cd"].size()[0]):
        for worker in range(data["cd"].size()[2]):
            # do fourier analysis for cd- and trajectories for each trajectory within the current episode
            f_cd, a_cd = welch(data["cd"][episode, :, worker] - pt.mean(data["cd"][episode, :, worker]),
                               fs=sampling_freq, nperseg=int(len_traj * 0.5), nfft=len_traj)
            f_cl, a_cl = welch(data["cl"][episode, :, worker] - pt.mean(data["cl"][episode, :, worker]),
                               fs=sampling_freq, nperseg=int(len_traj * 0.5), nfft=len_traj)
            freq_cd[episode, worker, :] = pt.tensor(f_cd)
            amp_cd[:, episode, worker] = pt.tensor(a_cd)
            freq_cl[episode, worker, :] = pt.tensor(f_cl)
            amp_cl[:, episode, worker] = pt.tensor(a_cl)

    # plot the mean amplitudes vs. frequencies and episodes and their corresponding standard deviation
    fig, ax = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
    p1 = ax[0][0].contourf(range(1, freq_cd.size()[0] + 1), range(freq_cd.size()[2]), pt.mean(amp_cd, dim=2) * 1000,
                           vmin=0, vmax=pt.max(pt.mean(amp_cd, dim=2) * 1000), levels=15)
    ax[0][1].contourf(range(1, freq_cd.size()[0] + 1), range(freq_cd.size()[2]), pt.std(amp_cd, dim=2) * 1000,
                      vmin=0, vmax=pt.max(pt.mean(amp_cd, dim=2) * 1000), levels=15)
    p2 = ax[1][0].contourf(range(1, freq_cl.size()[0] + 1), range(freq_cl.size()[2]), pt.mean(amp_cl, dim=2),
                           vmin=0, vmax=pt.max(pt.mean(amp_cl, dim=2)), levels=15)
    ax[1][1].contourf(range(1, freq_cl.size()[0] + 1), range(freq_cl.size()[2]), pt.std(amp_cl, dim=2),
                      vmin=0, vmax=pt.max(pt.mean(amp_cl, dim=2)), levels=15)

    ax[1][0].set_xlabel("$episode$ $number$", usetex=True, fontsize=13, labelpad=10)
    ax[1][1].set_xlabel("$episode$ $number$", usetex=True, fontsize=13, labelpad=10)
    ax[0][0].set_ylabel("$frequency$ $\qquad\left[\\frac{1}{100~epochs} \\right]$", usetex=True, fontsize=13,
                        labelpad=15)
    ax[1][0].set_ylabel("$frequency$ $\qquad\left[\\frac{1}{100~epochs} \\right]$", usetex=True, fontsize=13,
                        labelpad=15)
    ax[0][0].set_ylim(0, 25)
    ax[1][0].set_ylim(0, 25)
    ax[0][0].set_title("$mean$ $amplitudes$", usetex=True, fontsize=14)
    ax[0][1].set_title("$standard$ $deviation$", usetex=True, fontsize=14)
    cb1 = fig.colorbar(p1, ax=ax[0][1], shrink=0.75, format="%.2f")
    cb2 = fig.colorbar(p2, ax=ax[1][1], shrink=0.75, format="%.2f")
    cb1.set_label("$PSD$ $\qquad[10^{-3}]$", usetex=True, labelpad=20, fontsize=13)
    cb2.set_label("$PSD$ $\qquad[-]$", usetex=True, labelpad=20, fontsize=13)
    fig.tight_layout()
    plt.savefig(settings["main_load_path"] + settings["path_controlled"] + f"/plots/freq_vs_episodes_case{case}.png",
                dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def analyze_frequencies_final_result(settings: dict, uncontrolled_case: Union[dict, pd.DataFrame],
                                     controlled_case: list[Union[dict, pd.DataFrame]]) -> None:
    """
    analyzes the frequency spectrum of the trajectories for cd and cl of the final results (cases with active flow
    control using the final policies)

    :param settings: setup containing all the paths etc.
    :param uncontrolled_case: reference case containing results from uncontrolled flow past cylinder
    :param controlled_case: results from the loaded cases with active flow control using the final policies
    :return: None
    """
    # do frequency analysis for the trajectories of cl and cd for each loaded case
    f_cd, f_cl, a_cd, a_cl = [], [], [], []
    for case in range(len(settings["case_name"]) + 1):
        if case == 0:
            sampling_freq = 1 / (uncontrolled_case["t"][1] - uncontrolled_case["t"][0])
            len_traj = len(uncontrolled_case["t"])
            f_cd_tmp, a_cd_tmp = welch(uncontrolled_case["cd"] - uncontrolled_case["cd"].mean(), fs=sampling_freq,
                                       nperseg=int(len_traj * 0.5), nfft=len_traj)
            f_cl_tmp, a_cl_tmp = welch(uncontrolled_case["cl"] - uncontrolled_case["cl"].mean(), fs=sampling_freq,
                                       nperseg=int(len_traj * 0.5), nfft=len_traj)
        else:
            sampling_freq = 1 / (controlled_case[case - 1]["t"][1] - controlled_case[case - 1]["t"][0])
            len_traj = len(controlled_case[case - 1]["t"])
            f_cd_tmp, a_cd_tmp = welch(controlled_case[case - 1]["cd"] - controlled_case[case - 1]["cd"].mean(),
                                       fs=sampling_freq, nperseg=int(len_traj * 0.5), nfft=len_traj)
            f_cl_tmp, a_cl_tmp = welch(controlled_case[case - 1]["cl"] - controlled_case[case - 1]["cl"].mean(),
                                       fs=sampling_freq, nperseg=int(len_traj * 0.5), nfft=len_traj)
        f_cd.append(f_cd_tmp)
        f_cl.append(f_cl_tmp)
        a_cd.append(a_cd_tmp)
        a_cl.append(a_cl_tmp)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for i in range(2):
        for case in range(len(f_cd)):
            if i == 0:
                if case == 0:
                    ax[i].plot(f_cd[case], a_cd[case], color="black", label="uncontrolled")
                else:
                    ax[i].plot(f_cd[case], a_cd[case], color=settings["color"][case-1], label=settings["legend"][case-1])
                ax[i].set_xlabel("$f(c_d)$ $\quad[Hz]$", usetex=True, fontsize=13)
                ax[i].set_ylabel("$PDS(c_d)$ $\quad[-]$", usetex=True, fontsize=13)
                ax[i].set_xlim(0, 8)
            else:
                if case == 0:
                    ax[i].plot(f_cl[case], a_cl[case], color="black")
                else:
                    ax[i].plot(f_cl[case], a_cl[case], color=settings["color"][case-1])
                ax[i].set_xlabel("$f(c_l)$ $\quad[Hz]$", usetex=True, fontsize=13)
                ax[i].set_ylabel("$PDS(c_l)$ $\quad[-]$", usetex=True, fontsize=13)
                ax[i].set_xlim(0, 8)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, top=0.93)
    fig.legend(ncol=len(f_cd), loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig(settings["main_load_path"] + settings["path_controlled"] + f"/plots/freq_final_result.png", dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    pass
