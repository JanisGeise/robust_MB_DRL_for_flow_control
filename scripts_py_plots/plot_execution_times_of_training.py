"""
get the execution times for CFD, model training, PPO-training, MB-episodes and other from log file & plot the results
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join


def average_exec_times_over_seeds(path: str) -> dict:
    cases = ["time per CFD episode", "time per model training", "time per MB-episode",
             "time per update of PPO-agent", "other"]
    times = {"cfd": [], "mb": [], "model_train": [], "ppo": [], "other": [], "t_total": []}

    for seed in sorted(glob(path + "*.log"), key=lambda x: int(x.split(".")[0][-1])):
        with open(seed, "r") as f:
            data = f.readlines()

        for idx, key in enumerate(times):
            if key == "other":
                tmp = [data[line] for line in range(len(data)) if data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))

            elif key == "t_total":
                # get the total execution time of training
                times[key].append(float(data[-1].split()[-1].strip("\n")))

            else:
                tmp = [data[line + 5] for line in range(len(data) - 5) if data[line].startswith(cases[idx])]
                times[key].append(float(tmp[0].split(" ")[1]))

    # average exec times over all seeds -> note: due to avg. the times will not exactly add up to 100%
    for key in times:
        times[key] = pt.mean(pt.tensor(times[key]))

    return times


if __name__ == "__main__":
    setup = {
        "path": r"/home/janis/Hiwi_ISM/results_drlfoam_MB/run/final_routine/",
        "trainings": ["e80_r10_b10_f10_MB_1model/", "e80_r10_b10_f10_MB_5models_split_between_all_models/",
                      "e80_r10_b10_f10_MB_10models_split_between_all_models/"],
                      # "e80_r10_b10_f10_MB_20models_split_between_all_models/"],
        "labels": ["$N_{models} = 1$", "$N_{models} = 5$", "$N_{models} = 10$", "$N_{models} = 20$"],
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
            if key != "t_total":
                if idx == 0:
                    b = ax.bar(setup["labels"][idx], times[key], color=setup["color"][l], label=setup["legend"][l], bottom=bot)
                else:
                    b = ax.bar(setup["labels"][idx], times[key], color=setup["color"][l], bottom=bot)

                if key != "ppo" and key != "other":
                    t = "{:.2f} min".format((times[key] / 100) * times["t_total"] / 60)
                    ax.bar_label(b, label_type="center", labels=[t])
                bot += times[key]
    ax.set_ylabel("avg. execution time [\%]", usetex=True, fontsize=13)

    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=5)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig(join(setup["path"], "plots", "exec_times_relative.png"), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # plot the avg. exec time in [s]
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, times in enumerate(t_exec):
        bot = 0
        for l, key in enumerate(times):
            if key != "t_total":
                if idx == 0:
                    ax.bar(setup["labels"][idx], (times[key] / 100) * times["t_total"] / 60, color=setup["color"][l],
                           label=setup["legend"][l], bottom=bot)
                else:
                    ax.bar(setup["labels"][idx], (times[key] / 100) * times["t_total"] / 60, color=setup["color"][l],
                           bottom=bot)
                bot += ((times[key] / 100) * times["t_total"] / 60)
    ax.set_ylabel("avg. execution time [min]", usetex=True, fontsize=13)

    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=5)
    fig.subplots_adjust(wspace=0.25, top=0.93)
    plt.savefig(join(setup["path"], "plots", "exec_times_absolute.png"), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
