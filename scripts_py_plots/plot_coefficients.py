"""
    this script:
        - import and plots cl- and cd-coefficient (or other available coefficients) of the controlled cylinder flow
          using the final / trained policy in comparison with uncontrolled flow
"""
import pandas as pd
from math import ceil
from regex import findall, MULTILINE
import matplotlib.pyplot as plt


def get_reynolds_number(path: str, uncontrolled=True) -> float:
    """
    :param path: path to the simulation
    :param uncontrolled: flag whether controlled or uncontrolled simulation
    :return: Reynolds number used in the simulation rounded to the closest int
    """
    # path to dicts differs depending on uncontrolled or controlled case
    if uncontrolled:
        path_dict = r""
    else:
        path_dict = r"env/base_case/test_policy/"

    # in control dict: U_inf, lref, rho_inf
    with open(path + path_dict + r"system/controlDict", "r") as f:
        lines = f.read()
        re_data = findall(r"\s+magUInf\s+\d+.\d+;\s+lRef\s+\d+.\d+;", lines, MULTILINE)[0]
        re_data += findall(r"\s+rhoInf\s+\d+;", lines)[0]

    # get dynamic viscosity form transportProperties
    with open(path + path_dict + r"constant/transportProperties", "r") as f:
        nu = findall(r"nu\s+\d+.\d+e-\d+;", f.read())[0]

    # remove everything but the values and convert to floats
    re_data_stripped = [float(j) for j in findall(r"(\d+(?:\.\d+)?)", re_data)]
    return int(re_data_stripped[0] * re_data_stripped[1] * re_data_stripped[2] / float(findall(r"\d+.\d+e-\d+", nu)[0]))


def get_t_control_start(path: str) -> float:
    """
    :param path: path to the base_case simulation
    :return: start of the active flow control in [s]
    """
    # starting time is defined in controlDict.c as "timeStart"
    pattern = r"\s+timeStart\s+\d+.\d+;"
    with open(path + r"env/base_case/test_policy/system/controlDict", "r") as f:
        t = findall(pattern, f.read())[0]

    # remove everything but the value for timeStart
    return float(findall(r"\d+.\d+", t)[0])


if __name__ == "__main__":
    # Setup, 1st case is always assumed to be the uncontrolled reference case
    setup = {
        "main_path": r"/media/janis/Daten/Studienarbeit/run/",
        "path_cases": [r"cylinder2D_uncontrolled/cylinder2D_uncontrolled_Re100/",
                       "UE11/influenceManualSeed/drl_control_cylinder_seed0/"],
        "save_path": "UE11/influenceManualSeed/plots/",
        "path_coeffs": r"postProcessing/forces/0/coefficient.dat",
        "title": "UE 11: comparison uncontrolled vs. controlled case",
        "colors": ["black", "blue", "green", "red", "darkviolet", "magenta"],
        "legend_entries": ["uncontrolled case", "controlled case (seed = 0)"]
    }

    # get Renolds number of uncontrolled case, assuming same Re for the controlled cases (for comparison)
    setup["Re"] = get_reynolds_number(setup["main_path"] + setup["path_cases"][0])
    setup["title"] += f", $Re = {setup['Re']}$"

    data, t_start = [], []
    for i in range(len(setup["path_cases"])):
        # for controlled cases, path to coefficient.dat is different
        if i > 0:
            setup["path_coeffs"] = r"env/base_case/test_policy/postProcessing/forces/0/coefficient.dat"
        data.append(pd.read_csv(setup["main_path"] + setup["path_cases"][i] + setup["path_coeffs"], skiprows=13,
                                header=0, sep=r"\s+", usecols=[0, 1, 3], names=["t", "cd", "cl"]))

    # get starting times of active flow control for the controlled cases, 1st case is assumed to be the uncontrolled
    for i in range(1, len(setup["path_cases"])):
        t_start.append(get_t_control_start(setup["main_path"] + setup["path_cases"][i]))

    # plot coefficients and start for control times
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(len(data)):
        axes[0].plot(data[i]["t"], data[i]["cl"], color=setup["colors"][i], label=setup["legend_entries"][i])
        axes[1].plot(data[i]["t"], data[i]["cd"], color=setup["colors"][i])

        if i > 0:
            axes[0].vlines(t_start[i - 1], ceil(min(data[0]["cl"])), ceil(max(data[0]["cl"])), color="red",
                           linestyle="-.", lw=2)
            axes[1].vlines(t_start[i - 1], 2, 4, color="red", linestyle="-.", lw=2, label="start control")

    axes[0].set_xlabel("t [s]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_ylabel("lift coefficient \t$c_l$")
    axes[1].set_ylabel("drag coefficient \t$c_d$")
    axes[1].set_ylim(2.9, 3.25)
    fig.suptitle(setup["title"])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=2)
    plt.savefig(setup["main_path"] + setup["save_path"] + f"cl_cd_finalPolicy_Re{setup['Re']}.png", dpi=600)
    plt.show()
