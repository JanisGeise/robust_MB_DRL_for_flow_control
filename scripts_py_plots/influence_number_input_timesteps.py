"""
    brief:
        run parameter study to get the total prediction error of the probes, cd and cl depending on the number of time
        steps used as input for the environment model(s)

    dependencies:
        - 'train_environment_model.py'
        - 'post_process_results_env_model.py' for calculating the prediction error

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
from train_environment_model import *
from post_process_results_env_model import calculate_error_norm


def parameter_study_wrapper(settings: dict, trajectories: dict, n_time_steps: int, counter: int) -> pt.Tensor:
    """
    executes training-, validation and testing of an environment model(s) as well as the calculation of the prediction
    error

    :param settings: setup containing all the paths etc.
    :param trajectories: sampled trajectories in the CFD environment split into training-, validation and test data
    :param n_time_steps: number of time steps used as input
    :param counter: overall number of process to keep the calculations in separate directories
    :return: list containing L2- and L1-loss wrt the network architecture as
             [(neurons, layers), [[L2-losses], [L1-losses]]
    """
    print(f"parameter {counter}: starting calculation for network with {n_time_steps} time steps as input")
    settings["n_input_steps"] = n_time_steps

    if setup["episode_depending_model"]:
        if setup["two_env_models"]:
            predictions, _, _ = env_model_episode_wise_2models(settings, trajectories, n_neurons=settings["n_neurons"],
                                                               n_layers=settings["n_layers"], epochs=settings["epochs"])

        else:
            predictions, _, _ = train_test_env_model_episode_wise(settings, trajectories,
                                                                  n_neurons=settings["n_neurons"],
                                                                  n_layers=settings["n_layers"],
                                                                  epochs=settings["epochs"])

        # calculate L2- and L1-loss for the last episode, because previous two episodes are just used as training data
        loss = calculate_error_norm(predictions[0], trajectories["cl"][2, :, :], trajectories["cd"][2, :, :],
                                    trajectories["states"][2, :, :, :])

    else:
        if setup["two_env_models"]:
            predictions, _, _ = env_model_2models(settings, trajectories, n_neurons=settings["n_neurons"],
                                                  n_layers=settings["n_layers"], epochs=settings["epochs"])

        else:
            predictions, _, _ = train_test_env_model(settings, trajectories, n_neurons=settings["n_neurons"],
                                                     n_layers=settings["n_layers"], epochs=settings["epochs"])

        # calculate L2- and L1-loss for the current N input time steps based on predicted trajectories
        loss = calculate_error_norm(predictions, trajectories["cl_test"], trajectories["cd_test"],
                                    trajectories["states_test"])

    return loss

    return loss


def manage_network_training_single_proc(settings: dict, trajectory_data: dict) -> list:
    """
    manages the execution of the parameter study with single processes

    :param settings: setup containing all the path etc.
    :param trajectory_data: sampled trajectories in the CFD environment split into training-, validation and test data
    :return: list containing the L2- and L1-norm of the error for each input time step, the data is stored as
             [[L2-norm states, L2-norm cl, L2-norm cd], [L1-norm states, L1-norm cl, L1-norm cd]]
    """
    if not os.path.exists(settings["load_path"] + settings["model_dir"]):
        os.mkdir(settings["load_path"] + settings["model_dir"])

    # save some parameters, because they get overwritten later
    res, params = [], settings["n_input_steps"]
    for counter in range(len(settings["n_input_steps"])):
        res.append(parameter_study_wrapper(settings, trajectory_data, params[counter], counter))

        # clean up the directory after each iteration
        for file in glob("".join([settings["load_path"], settings["model_dir"], "/*.pt"])):
            os.remove(file)
    return res


def create_subset_of_data(data: dict, episode_no: int = 2) -> dict:
    """
    if new model should be trained every new episode, only do parameter study for the episode defined in the setup,
    because running a parameter study for all episodes would be computationally to demanding (also takes too much time)

    :param data: the loaded data containing all trajectories
    :param episode_no: for which episode should the parameter study be done
    :return: the trajectories of the two episodes before the target episode as training data, and trajectories of the
             target episode as test data
    """
    data_new = {}
    trajectory_keys = ["actions", "cl", "cd", "states"]
    for key in data:
        if key in trajectory_keys:
            data_new[key] = data[key][episode_no-2:episode_no+1, :, :]
        else:
            data_new[key] = data[key]

    return data_new


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"/media/janis/Daten/Studienarbeit/drlfoam/examples/test_training3/",
        "path_to_probes": r"base/postProcessing/probes/0/",
        "model_dir": "Results_model/influence_input_timesteps/two_env_models/one_model_for_all_data/cd_filtered",
        "episode_depending_model": False,   # either one model for whole data set or new model for each episode
        "which_episode": 2,                 # for which episode should the parameter study be done (1st episode = zero)
        "two_env_models": False,            # 'True': one model only for predicting cd, another for probes and cl
        "print_temp": False,                # print core temperatur of processor as info
        "normalize": True,                  # True: data will be normalized to interval of [1, 0]
        "n_input_steps": [2, 3, 5, 10, 15, 20, 25, 30, 35],  # number of initial time steps as input
        "len_trajectory": 200,              # trajectory length for training the environment model
        "ratio": (0.65, 0.3, 0.05),         # splitting ratio for train-, validation and test data
        "smooth_cd": True,                  # flag if cd-Trajectories should be filtered after loading (low-pass filter)
        "epochs": 10000,                    # number of epochs to run for the environment model
        "n_neurons": 50,                    # number of neurons per layer for the environment model
        "n_layers": 3,                      # number of hidden layers for the environment model
        "n_neurons_cd": 50,                 # number of neurons per layer for the env model for cd (if option is set)
        "n_layers_cd": 5,                   # number of hidden layers for the env model for cd (if option is set)
        "epochs_cd": 10000,                 # number of epochs to run for the env model for cd (if option is set)
    }
    # save number of time steps computed for later, because gets overwritten
    x = pt.tensor(setup["n_input_steps"])

    if setup["episode_depending_model"]:
        assert setup["ratio"][2] == 0, "for episode depending model the test data ratio must be set to zero!"

    # load the sampled trajectories divided into training-, validation- and test data
    pt.Generator().manual_seed(0)                           # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    if setup["episode_depending_model"]:
        # only train and test model for the defined episode in the setup, because all episodes at once not feasible
        data_for_one_episode = create_subset_of_data(divided_data, setup["which_episode"])
        losses = manage_network_training_single_proc(setup, data_for_one_episode)
    else:
        losses = manage_network_training_single_proc(setup, divided_data)

    # plot L1- and L2-norm of error made by the environment model(s)
    loss = pt.stack(losses, dim=0)
    for norm in range(2):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.plot(x, loss[:, norm, 0], color="red", label="$probes$", marker="o", fillstyle="none")
        ax.plot(x, loss[:, norm, 1], color="blue", label="$c_l$", marker="o", fillstyle="none")
        ax.plot(x, loss[:, norm, 2], color="green", label="$c_d$", marker="o", fillstyle="none")
        plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=3)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("$number$ $of$ $time$ $steps$ $as$ $input$", usetex=True, fontsize=13)
        if norm == 0:
            ax.set_ylabel("$L_2-norm$ $(relative$ $prediction$ $error)$", usetex=True, fontsize=12)
        else:
            ax.set_ylabel("$L_1-norm$ $(relative$ $prediction$ $error)$", usetex=True, fontsize=12)
        fig.subplots_adjust(left=0.04)
        fig.tight_layout()
        if norm == 0:
            plt.savefig("".join([setup["load_path"], setup["model_dir"], "/total_prediction_error_L2norm.png"]),
                        dpi=600)
        else:
            plt.savefig("".join([setup["load_path"], setup["model_dir"], "/total_prediction_error_L1norm.png"]),
                        dpi=600)
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
