"""
get the MF- and MB-episodes of an MB-training and plot their distribution wrt episodes
"""

import torch as pt
import matplotlib.pyplot as plt

from glob import glob


if __name__ == "__main__":
    path = r"/home/janis/Hiwi_ISM/results_drlfoam_MB/run/new_switching_criteria/"
    case = r"e80_r10_b10_f8_MB_5models_threshold_60/"
    mf_episodes, n_episodes = [], []

    for seed in sorted(glob(path + case + "*.log"), key=lambda x: int(x.split(".")[0][-1])):
        with open(seed, "r") as f:
            data = f.readlines()
        mf_episodes.append([data[line] for line in range(len(data)-1) if
                            data[line].startswith("Start of episode ") and data[line+1].startswith("Runner ")])
        n_episodes.append(int([data[line] for line in range(len(data)) if
                          data[line].startswith("Start of episode ")][-1].split(" ")[-1].strip("\n")))

    # get the number of the MF-episodes and MB-episodes
    mf_episodes = [[int(e.split(" ")[-1].strip("\n")) for e in c] for c in mf_episodes]
    mb_episodes = [[e for e in range(n_episodes[idx] + 1) if e not in mf] for idx, mf in enumerate(mf_episodes)]

    # plot MF & MB episodes vs. seed / training
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    for s in range(len(mf_episodes)):
        if s == 0:
            ax.plot(mf_episodes[s], pt.ones(len(mf_episodes[s])) * s, label=f"CFD episodes", ls="None", marker="o",
                    color="red")
            ax.plot(mb_episodes[s], pt.ones(len(mb_episodes[s])) * s, label=f"MB episodes", ls="None", marker="o",
                    color="green")
        else:
            ax.plot(mf_episodes[s], pt.ones(len(mf_episodes[s])) * s, ls="None", marker="o", color="red")
            ax.plot(mb_episodes[s], pt.ones(len(mb_episodes[s])) * s, ls="None", marker="o", color="green")
    ax.set_yticks(range(len(mf_episodes)))
    ax.set_xlabel("$episode$ $number$", usetex=True, fontsize=13)
    ax.set_ylabel("$case$ $number$", usetex=True, fontsize=13)
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig("".join([path, case, "/mf_mb_episodes.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
