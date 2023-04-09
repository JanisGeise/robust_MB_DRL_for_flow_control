"""
get the execution times for CFD, model training, PPO-training, MB-episodes and other from log file & plot the results
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join


def average_exec_times_over_seeds(path: str) -> dict:
    cases = [b"time per CFD episode", b"time per model training", b"time per MB-episode",
             b"time per update of PPO-agent", b"other"]
    times = {"cfd": [], "mb": [], "model_train": [], "ppo": [], "other": []}

    for seed in sorted(glob(path + "*.log"), key=lambda x: int(x.split(".")[0][-1])):
        with open(seed, "rb") as f:
            data = f.readlines()

        for idx, key in enumerate(times):
            if key != "other":
                tmp = [data[line + 5].decode("utf-8") for line in range(len(data) - 5) if
                       data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))
            else:
                tmp = [data[line].decode("utf-8") for line in range(len(data)) if data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))

    # average exec times over all seeds -> note: due to avg. the times will not exactly add up to 100%
    for key in times:
        times[key] = pt.mean(pt.tensor(times[key]))

    return times


if __name__ == "__main__":
    setup = {
        "path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/run/new_switching_criteria/",
        "trainings": [r"e80_r10_b10_f6_MB_5models_CPU_new_training/", r"e80_r10_b10_f6_MB_10models_CPU_new_training/",
                      r"e80_r10_b10_f6_MB_20models_CPU_new_training/"],
        "labels": ["$N_{models} = 5$", "$N_{models} = 10$", "$N_{models} = 20$"],
        "legend": ["CFD episode", "MB episode", "model training", "PPO training", "other"],
        "color": ["red", "green", "blue", "darkviolet", "black"],
    }
    t_exec = []

    for t in setup["trainings"]:
        t_exec.append(average_exec_times_over_seeds(join(setup["path"], t)))

    # plot the avg. exec time in percent
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, times in enumerate(t_exec):
        bot = 0
        for l, key in enumerate(times):
            if idx == 0:
                ax.bar(setup["labels"][idx], times[key], color=setup["color"][l], label=setup["legend"][l], bottom=bot)
            else:
                ax.bar(setup["labels"][idx], times[key], color=setup["color"][l], bottom=bot)
            bot += times[key]
    ax.set_ylabel("avg. execution time [\%]", usetex=True, fontsize=13)

    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=5)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig(join(setup["path"], "plots", "composition_exec_times.png"), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
