"""
    brief:
        - tests the influence of number of neurons and hidden layers on results of the predictions of the environment
          model
        - plots the resulting L2- and L1-norm of the error within the test data set as heatmaps for the predicted
          states, cl and cd

    dependencies:
        - 'trainModel.py'

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import seaborn as sns
from train_environment_model import *


def calculate_error_norm(pred_trajectories: list, cl_test: pt.Tensor, cd_test: pt.Tensor,
                         states_test: pt.Tensor) -> pt.Tensor:
    """
    :brief: calculates the L2- and L1-norm of the error between the real and the predicted trajectories within the test
            data set
    :param pred_trajectories: predicted trajectories by the environment model
    :param cl_test: cl coefficients of the test data set sampled in the CFD environment
    :param cd_test: cd coefficients of the test data set sampled in the CFD environment
    :param states_test: states at the probe locations of the test data set sampled in the CFD environment
    :return: tensor containing the L2- and L1-norm of the error, the data is stored as
             [[L2-norm states, L2-norm cl, L2-norm cd], [L1-norm states, L1-norm cl, L1-norm cd]]
    """
    # resort prediction data from list of dict to tensors, so that they have the same shape as the test data
    pred_states, pred_cl, pred_cd = pt.zeros(states_test.size()), pt.zeros(cl_test.size()), pt.zeros(cd_test.size())
    for trajectory in range(len(pred_trajectories)):
        pred_states[:, :, trajectory] = pred_trajectories[trajectory]["states"]
        pred_cl[:, trajectory] = pred_trajectories[trajectory]["cl"]
        pred_cd[:, trajectory] = pred_trajectories[trajectory]["cd"]

    # calculate MSE (L2) and L1 loss
    l2 = pt.nn.MSELoss()
    l1 = pt.nn.L1Loss()
    all_losses = [(l2(states_test, pred_states).item(), l2(cl_test, pred_cl).item(), l2(cd_test, pred_cd).item()),
                  (l1(states_test, pred_states).item(), l1(cl_test, pred_cl).item(), l1(cd_test, pred_cd).item())]

    return pt.tensor(all_losses)


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
    # fig.suptitle(f"{parameter}")
    fig.tight_layout()
    plt.savefig(settings["load_path"] + settings["model_dir"] +
                f"/plots/l2_l1_error_vs_model_parameters_{parameter}.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"/media/janis/Daten/Studienarbeit/drlfoam/examples/test_training3/",
        "model_dir": "Results_model/normalized_data/influenceModelArchitecture",  # name of directory for plots
        "normalize": True,                                  # True: input data will be normalized to interval of [1, 0]
        "n_input_steps": 2,                                 # initial time steps as input -> n_input_steps > 1
        "len_trajectory": 200,                              # trajectory length for training the environment model
        "ratio": (0.65, 0.3, 0.05),                         # splitting ratio for train-, validation and test data
        "n_neurons": [10, 25, 50, 100, 200],                # number of neurons per layer which should be tested
        "n_layers": [2, 3, 5, 10, 15]                       # number of hidden layers which should be tested
    }

    # load the sampled trajectories divided into training-, validation- and test data
    pt.Generator().manual_seed(0)  # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    # allocate tensor for storing the L1- and L2 loss of states, cl and cd for each neuron-layer combination
    losses = pt.zeros((len(setup["n_neurons"]), len(setup["n_layers"]), 2, 3))

    # loop over neuron- and hidden layer-list
    for neurons in range(len(setup["n_neurons"])):
        for layers in range(len(setup["n_layers"])):
            predictions, _, _ = run_parameter_study(setup, divided_data, n_neurons=setup["n_neurons"][neurons],
                                                    n_layers=setup["n_layers"][layers], epochs=10000)

            # calculate L2- and L1-loss for each neuron-layer combination based on predicted trajectories
            losses[neurons, layers, :, :] = calculate_error_norm(predictions, divided_data["cl_test"],
                                                                 divided_data["cd_test"], divided_data["states_test"])

            print(f"finished calculation for network with {setup['n_neurons'][neurons]} neurons and"
                  f" {setup['n_layers'][layers]} layers")

    if not os.path.exists(setup["load_path"] + setup["model_dir"] + "/plots"):
        os.mkdir(setup["load_path"] + setup["model_dir"] + "/plots")

    # plot L2 and L1 losses wrt n_neurons and n_layers as heatmap
    plot_losses(setup, losses, parameter="states")
    plot_losses(setup, losses, parameter="cl")
    plot_losses(setup, losses, parameter="cd")