"""
    brief:
        - tests the influence of number of neurons and hidden layers on results of the predictions of the environment
          model(s)
        - plots the resulting L2- and L1-norm of the error within the test data set as heatmaps for the predicted
          states, cl and cd
        - Note: this script uses the same training routine as implemented in 'mb_drl/env_model_rotating_cylinder_old_routine.py'

    dependencies:
        - 'train_environment_model.py'

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import seaborn as sns
from shutil import rmtree

from train_environment_model import *
from influence_number_input_timesteps import create_subset_of_data


def parameter_study_wrapper(settings: dict, trajectories: dict, network_architecture: tuple, counter: int) -> list:
    """
    :brief: executes training-, validation and testing of an environment model as well as the loss calculation
    :param settings: setup containing all the paths etc.
    :param trajectories: sampled trajectories in the CFD environment split into training-, validation and test data
    :param network_architecture: tuple containing the number of neurons and hidden layers as (neurons, layers)
    :param counter: overall number of process to keep the calculations in separate directories
    :return: list containing L2- and L1-loss wrt the network architecture as
             [(neurons, layers), [[L2-losses], [L1-losses]]
    """
    print(f"process {counter}: starting calculation for network with {network_architecture[0]} neurons and "
          f"{network_architecture[1]} layers")

    # make temporary directory for each process
    settings["model_dir"] += f"/tmp_no_{counter}"

    # initialize and train environment network, test best model
    predictions, _, _ = train_test_env_model(setup, divided_data, n_neurons=setup["n_neurons"],
                                             n_layers=setup["n_layers"], epochs=setup["epochs"])

    # calculate L2- and L1-loss for each neuron-layer combination based on predicted trajectories
    loss = calculate_error_norm(predictions, trajectories["cl_test"], trajectories["cd_test"],
                                trajectories["states_test"])

    # delete tmp directory and everything in it
    rmtree(settings["load_path"] + settings["model_dir"])

    return [network_architecture, loss]


def plot_losses(settings: dict, loss_data: pt.Tensor, parameter: str = "cd") -> None:
    """
    :brief: creates heatmaps of the L2- and L1-losses wrt n_neurons and n_layers
            note:
                - seaborn plots row-wise (in x-direction) starting in the top-left corner
                    -> x-data is n_layers, y-data is n_neurons
    :param settings: setup defining paths etc.
    :param loss_data: the tensor containing the calculated L2- and L1-losses for each neuron-layer combination
    :param parameter: which losses (CFD vs. prediction by the environment model) of which parameter should be plotted:
            1) 'states': plots losses of the states measured at the probe locations within the flow field
            2) 'cl': plots losses of the cl-coefficient at the cylinder surface
            3) 'cd': plots losses of the cd-coefficient at the cylinder surface
    :return: None
    """
    if parameter == "states":
        idx = 0
    elif parameter == "cl":
        idx = 1
    elif parameter == "cd":
        idx = 2
    else:
        idx = 42
        print("specify which parameter to plot, either: 'states', 'cl' or 'cd'")
        exit()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(2):
        if i == 0:
            vmin = float("{:.4f}".format(pt.min(loss_data[:, :, i, idx]).item()))
            vmax = float("{:.4f}".format(pt.max(loss_data[:, :, i, idx]).item()))
            ax[i].set_title("$L_2-norm$", usetex=True, fontsize=16)
        else:
            vmin = float("{:.3f}".format(pt.min(loss_data[:, :, i, idx]).item()))
            vmax = float("{:.3f}".format(pt.max(loss_data[:, :, i, idx]).item()))
            ax[i].set_title("$L_1-norm$", usetex=True, fontsize=16)

        sns.heatmap(loss_data[:, :, i, idx], vmin=vmin, vmax=vmax, center=0, cmap="Greens", square=True, annot=True,
                    cbar=True, linewidths=0.30, linecolor="white", ax=ax[i], xticklabels=settings["n_layers"],
                    yticklabels=settings["n_neurons"], cbar_kws={"shrink": 0.75}, fmt=".4g")
        ax[i].set_xlabel("$N_{layers}$", usetex=True, fontsize=13)
        ax[i].set_ylabel("$N_{neurons}$", usetex=True, fontsize=13)

        # since seaborn starts plotting as top-left corner -> axis needs to be inverted
        ax[i].invert_yaxis()
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    if settings["study_cd_model"]:
        plt.savefig("".join([settings["load_path"], settings["model_dir"],
                    f"/plots/l2_l1_error_vs_model_parameters_{parameter}_cd_model.png"]), dpi=600)
    else:
        plt.savefig("".join([settings["load_path"], settings["model_dir"],
                    f"/plots/l2_l1_error_vs_model_parameters_{parameter}_cl_p_model.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def check_setup(settings: dict) -> None:
    """
    checks if there are invalid options set in the setup

    :param settings: setup for the parameter study
    :return: None
    """
    if settings["study_cd_model"]:
        assert settings["two_env_models"] is True, f"for doing a parameter study for the cd model, the option" \
                                                   f"'two_env_models' needs to be set to 'True'"
        assert type(settings["n_layers_cd"]) is list and type(settings["n_neurons_cd"]) is list, \
            "'n_neurons_cd' and 'n_layers_cd' need to be a list of integers"
        assert type(settings["n_layers"]) is not list and type(settings["n_neurons"]) is not list, \
            "'n_neurons' and 'n_layers' need to be integer values"
    else:
        assert type(settings["n_layers"]) is list and type(settings["n_neurons"]) is list, \
            "'n_neurons' and 'n_layers' need to be a list of integers"
        assert type(settings["n_layers_cd"]) is not list and type(settings["n_neurons_cd"]) is not list, \
            "'n_neurons_cd' and 'n_layers_cd' need to be integer values"

    if settings["episode_depending_model"]:
        assert settings["ratio"][2] == 0, "for episode depending model the test data ratio must be set to zero!"
    assert settings["n_input_steps"] > 1, f"setup['n_input_steps'] has to be > 1"


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"../drlfoam/examples/test_training/",     # path with the training data of drlfoam
        "path_to_probes": r"base/postProcessing/probes/0/",     # should always be the same
        "model_dir": "test_env_models/",    # relative to the load_path
        "episode_depending_model": True,   # either one model for whole data set or new model for each episode
        "which_episode": 2,                 # for which episode should the parameter study be done (1st episode = zero)
        "two_env_models": False,             # 'True': one model only for predicting cd, another for probes and cl
        "print_temp": False,                # print core temperatur of processor as info
        "normalize": True,                  # True: data will be normalized to interval of [1, 0]
        "smooth_cd": False,                 # flag if cd-Trajectories should be filtered after loading (low-pass filter)
        "predict_ds": True,                 # use change of state for prediction, not the next state (not recommended)
        "study_cd_model": False,            # 'True': do parameter study for cd-model, only available if 2 env. models
        "n_input_steps": 15,                # initial time steps as input
        "len_trajectory": 200,              # trajectory length for training the environment model
        "ratio": (0.65, 0.35, 0.0),         # splitting ratio for train-, validation and test data
        "epochs": 100,                    # number of epochs to run for the environment model
        "n_neurons": [25, 50],     # number of neurons per layer which should be tested
        "n_layers": [2, 3],           # number of hidden layers which should be tested
        "n_neurons_cd": 50,                 # number of neurons per layer for the env model for cd (if option is set)
        "n_layers_cd": 5,                   # number of hidden layers for the env model for cd (if option is set)
        "epochs_cd": 100,                 # number of epochs to run for the env model for cd (if option is set)
    }

    # check if there are issues with the current setup
    check_setup(setup)

    # load the sampled trajectories and divide them into training-, validation- and test data
    pt.manual_seed(0)                                   # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    # allocate tensor for storing the L1- and L2 loss of states, cl and cd for each neuron-layer combination
    losses = pt.zeros((len(setup["n_neurons"]), len(setup["n_layers"]), 2, 3))

    # depending on which option are chosen train and test environment model(s)
    if setup["two_env_models"]:
        if setup["episode_depending_model"]:
            train_fct = env_model_episode_wise_2models
            divided_data = create_subset_of_data(divided_data, setup["which_episode"])
        else:
            train_fct = env_model_2models
    else:
        if setup["episode_depending_model"]:
            train_fct = train_test_env_model_episode_wise
            divided_data = create_subset_of_data(divided_data, setup["which_episode"])
        else:
            train_fct = train_test_env_model

    # loop over neuron- and hidden layer-list
    for neurons in range(len(setup["n_neurons"])):
        for layers in range(len(setup["n_layers"])):
            # depending on for which model the parameter study should be conducted
            if setup["study_cd_model"]:
                n_neurons, n_layers = setup["n_neurons_cd"][neurons], setup["n_layers_cd"][neurons]
            else:
                n_neurons, n_layers = setup["n_neurons"][neurons], setup["n_layers"][neurons]

            print(f"staring calculation for network with {setup['n_neurons'][neurons]} neurons and"
                  f" {setup['n_layers'][layers]} layers...")

            predictions, _, _ = train_fct(setup, divided_data, n_neurons=n_neurons, n_layers=n_layers,
                                          epochs=setup["epochs"])

            # calculate L2- and L1-loss for each neuron-layer combination based on predicted trajectories
            if setup["episode_depending_model"]:
                # for computing the error norms of episode-depending model, there exist only one episode
                losses[neurons, layers, :, :] = calculate_error_norm(predictions[0], divided_data["cl"][-1, :, :],
                                                                     divided_data["cd"][-1, :, :],
                                                                     divided_data["states"][-1, :, :, :])
            else:
                losses[neurons, layers, :, :] = calculate_error_norm(predictions, divided_data["cl_test"],
                                                                     divided_data["cd_test"],
                                                                     divided_data["states_test"])

            print(f"finished calculation for network with {setup['n_neurons'][neurons]} neurons and"
                  f" {setup['n_layers'][layers]} layers")

    if not path.exists("".join([setup["load_path"] + setup["model_dir"] + "/plots"])):
        mkdir("".join([setup["load_path"] + setup["model_dir"] + "/plots"]))

    # save losses
    pt.save(losses, "".join([setup["load_path"], setup["model_dir"], "/losses.pt"]))

    # plot L2 and L1 losses wrt n_neurons and n_layers as heatmap
    plot_losses(setup, losses, parameter="states")
    plot_losses(setup, losses, parameter="cl")
    plot_losses(setup, losses, parameter="cd")
