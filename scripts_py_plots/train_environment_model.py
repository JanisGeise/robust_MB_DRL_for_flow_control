"""
    brief:
        - implements and trains a fully connected NN-model for approximating the CFD environment using the sampled
          trajectories as trainings-, validation- and test data
        - tries to predict trajectories based on a given initial state and actions

    dependencies:
        - None

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import glob
import os
import pickle
import torch as pt
import regex as re
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union


class FCModel(pt.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_neurons, activation: callable = pt.nn.functional.relu):
        """
        :param n_inputs: N probes * N time steps + 1 action + cl + cd
        :param n_outputs: N probes
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: none
        """
        super(FCModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.layers = pt.nn.ModuleList()

        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))

        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(
                    self.n_neurons, self.n_neurons))

        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)


def create_label_feature_pairs(t_idx: int, idx_trajectory: Union[pt.Tensor, int], n_time_steps, trajectory: pt.Tensor,
                               cl: pt.Tensor, cd: pt.Tensor, action: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    :brief: creates feature-label pairs for in- / output of environment model
    :param t_idx: index within the trajectory used as starting point for input states
    :param idx_trajectory: number of the trajectory within the whole training data set
    :param n_time_steps: number of time steps used for input
    :param trajectory: the trajectory used as feature / label
    :param cl: corresponding cl coefficient to the in- and output states
    :param cd: corresponding cd coefficient to the in- and output states
    :param action: actions taken in the input states
    :return: one tensor containing all the features containing all states, actions, cl- and cd-values for n_time_steps;
             one tensor containing the corresponding labels. The labels state the environment at the state
             (t_idx + n_time_steps + 1) while the features contain all states, action
    """
    # [n_probes * n_time_steps * states, n_time_steps * cl, n_time_steps * cd, n_time_steps * action]
    input_state = trajectory[t_idx:t_idx + n_time_steps, :].squeeze()
    c_l = cl[t_idx:t_idx + n_time_steps, idx_trajectory]
    c_d = cd[t_idx:t_idx + n_time_steps, idx_trajectory]
    action = action[t_idx:t_idx + n_time_steps, idx_trajectory]
    feature = pt.concat([input_state, c_l, c_d, action], dim=1)

    # labels contain next step to predict by the model
    label = pt.concat([trajectory[t_idx + n_time_steps, :].squeeze(), cl[t_idx + n_time_steps, idx_trajectory],
                       cd[t_idx + n_time_steps, idx_trajectory]], dim=0)

    return pt.flatten(feature), pt.flatten(label)


def train_model(model: pt.nn.Module, state_train: pt.Tensor, action_train: pt.Tensor, state_val: pt.Tensor,
                action_val: pt.Tensor, cl_train: pt.tensor, cd_train: pt.tensor, cl_val: pt.tensor, cd_val: pt.tensor,
                epochs: int = 10000, lr: float = 0.0005, batch_size: int = 50, n_time_steps: int = 3,
                save_model: bool = True, save_name: str = "best",
                save_dir: str = "EnvModel") -> Tuple[list[float], list[float]]:
    """
        :brief: train environment model based on sampled trajectories
        :param model: environment model
        :param state_train: states for training
        :param action_train: actions for training
        :param state_val: states for validation
        :param action_val: actions for validation
        :param cd_train: cd values for training
        :param cl_train: cl values for validation
        :param cd_val: cd values for validation
        :param cl_val: cl values for validation
        :param epochs: number of epochs for training
        :param lr: learning rate
        :param batch_size: batch size
        :param n_time_steps: number of input time steps
        :param save_model: option to save best model, default is True
        :param save_dir: path to directory where models should be saved
        :param save_name: name of the model saved, default is number of epoch
        :return: training and validation loss as list
        """

    assert state_train.shape[0] > n_time_steps + 1, "input number of time steps greater than trajectory length!"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    pt.autograd.set_detect_anomaly(True)

    # allocate some tensors for feature-label selection and losses
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    training_loss, validation_loss = [], []
    samples_train, samples_val = pt.ones(state_train.shape[-1]), pt.ones(state_val.shape[-1])
    samples_t_start = pt.ones(state_train.shape[0] - n_time_steps)

    for epoch in range(epochs):
        # randomly select a trajectory and starting points wrt the selected batch size out of the data sets
        idx_train, idx_val = pt.multinomial(samples_train, 1), pt.multinomial(samples_val, 1)
        t_start_idx = pt.multinomial(samples_t_start, batch_size)
        traj_train, traj_val = state_train[:, :, idx_train], state_val[:, :, idx_val]

        # train model
        batch_loss = pt.zeros(batch_size)
        for b in range(batch_size):
            model.train()
            optimizer.zero_grad()

            # create pairs with feature and corresponding label
            feature, label = create_label_feature_pairs(t_start_idx[b].item(), idx_train, n_time_steps, traj_train,
                                                        cl_train, cd_train, action_train)

            # get prediction and loss based on n time steps for next state
            prediction = model(feature).squeeze()
            loss_train = criterion(prediction, label)
            loss_train.backward()
            optimizer.step()
            batch_loss[b] = loss_train.item()

        training_loss.append(pt.mean(batch_loss).item())

        # validation loop
        with pt.no_grad():
            for b in range(batch_size):
                feature, label = create_label_feature_pairs(t_start_idx[b].item(), idx_val, n_time_steps, traj_val,
                                                            cl_val, cd_val, action_val)
                prediction = model(feature).squeeze()
                loss_val = criterion(prediction, label)
                batch_loss[b] = loss_val.item()

        validation_loss.append(pt.mean(batch_loss).item())

        # save model after every 250 epochs, also save best model
        if epoch % 250 == 0:
            pt.save(model.state_dict(), f"{save_dir}/model_training_epoch{epoch}.pt")

        if save_model:
            if training_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_dir}/{save_name}_train.pt")
                best_train_loss = training_loss[-1]
            if validation_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_dir}/{save_name}_val.pt")
                best_val_loss = validation_loss[-1]

        # print some info after every 50 epochs
        if epoch % 50 == 0:
            print(f"finished epoch {epoch}, training loss = {training_loss[epoch]}, "
                  f"validation loss = {validation_loss[epoch]}")

    return training_loss, validation_loss


def load_trajectory_data(path: str) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    :brief: load observations_*.pkl files containing all the data generated during training
    :param path: path to directory containing the files
    :return: actions, states, cl, cd as tensors where every column is a trajectory
    """
    # for training an environment model it doesn't matter in which order files are read in -> no sorting required
    files = glob.glob(path + "observations_*.pkl")
    observations = [pickle.load(open(file, "rb")) for file in files]

    """
    resort loaded data, because every epoch contains multiple trajectories from all workers but for training it doesn't
    matter in which order or from which worker the trajectory was created, here: assuming all trajectories have same
    length and same number of workers and same number of probes
    """
    n_traj = len(observations) * len(observations[0])
    len_traj = len(observations[0][0]["actions"])
    n_probes, n_col = len(observations[0][0]["states"][0]), 0
    states = pt.zeros((len_traj, n_probes, n_traj))
    actions = pt.zeros((len_traj, n_traj))
    cl = pt.zeros((len_traj, n_traj))
    cd = pt.zeros((len_traj, n_traj))

    for observation in range(len(observations)):
        for j in range(len(observations[observation])):
            actions[:, n_col] = observations[observation][j]["actions"]
            cl[:, n_col] = observations[observation][j]["cl"]
            cd[:, n_col] = observations[observation][j]["cd"]
            states[:, :, n_col] = observations[observation][j]["states"][:]
            n_col += 1
    return actions, states, cl, cd


def import_probe_locations(path: str) -> np.ndarray:
    """
    :brief: import probe locations of the tested base case
    :param path: path to the post-processing directory of the base case
    :return: x-, y- and z-coordinates of all probe locations as ndarray
    """
    pattern = r"\d.\d+ \d.\d+ \d.\d+"
    with open(path + "p", "r") as f:
        loc = f.readlines()

    # get coordinates of probes, omit appending empty lists and map strings to floats
    coord = [re.findall(pattern, line) for line in loc if re.findall(pattern, line)]
    positions = [list(map(float, i[0].split())) for i in coord]
    return np.array(positions)


def split_data(states: pt.Tensor, actions: pt.tensor, cl: pt.tensor, cd: pt.tensor, ratio: Tuple):
    """
    :brief: split trajectories into train-, validation and test data
    :param states: sampled states in CFD environment
    :param actions: sampled actions in CFD environment
    :param cd: sampled cl-coefficients in CFD environment (at cylinder surface)
    :param cl: sampled cd-coefficients in CFD environment (at cylinder surface)
    :param ratio: ratio between training-, validation- and test data as tuple as (train, validation, test)
    :return: dictionary with splitted states and corresponding actions
    """
    data = {}
    # split dataset into training data, validation data and testdata
    n_train, n_val, n_test = int(ratio[0] * actions.size()[1]), int(ratio[1] * actions.size()[1]), int(ratio[2] * actions.size()[1])

    # randomly select indices of trajectories
    samples = pt.ones(actions.shape[-1])
    idx_train = pt.multinomial(samples, n_train)
    idx_val = pt.multinomial(samples, n_val)
    idx_test = pt.multinomial(samples, n_test)

    # assign train-, validation and testing data based on chosen indices
    data["actions_train"], data["actions_val"], data["actions_test"] = actions[:, idx_train], actions[:, idx_val], \
                                                                       actions[:, idx_test]
    data["states_train"], data["states_val"], data["states_test"] = states[:, :, idx_train], states[:, :, idx_val], \
                                                                    states[:, :, idx_test]
    data["cl_train"], data["cd_train"] = cl[:, idx_train], cd[:, idx_train]
    data["cl_val"], data["cd_val"] = cl[:, idx_val], cd[:, idx_val]
    data["cl_test"], data["cd_test"] = cl[:, idx_test], cd[:, idx_test]

    return data


def predict_trajectories(model: pt.nn.Module, input: pt.Tensor, actions: pt.Tensor, n_probes: int) -> dict:
    """
    :brief: using the environment model in order to predict the trajectory based on a given initial state and actions
    :param model: NN model of the environment
    :param input: input states, cl, cd and actions for the first N time steps
    :param actions: actions taken in the CFD environment along the whole trajectory
    :param n_probes: number of probes used in the simulation
    :return: dictionary with the trajectories containing the states at the probe locations, cl, cd
    """
    # allocate tensor for predicted states, cl and cd for the whole trajectory and fill in the first N input states
    trajectory = pt.zeros([len(actions), input.size()[1] - 1])
    trajectory[:input.size()[0], :] = input[:input.size()[0], :-1]

    # loop over the trajectory
    for t in range(len(actions) - input.size()[0]):
        # make prediction and append to trajectory tensor, then move input window by one time step
        feature = pt.flatten(pt.concat([trajectory[t:t + input.size()[0], :],
                                        (actions[t:t + input.size()[0]]).reshape([input.size()[0], 1])], dim=1))
        trajectory[t + input.size()[0], :] = model(feature).squeeze()

    # resort: divide output data into states, cl and cd
    output = {"states": trajectory[:, :n_probes], "cl": trajectory[:, -2], "cd": trajectory[:, -1]}
    return output


def normalize_data(x: pt.Tensor) -> Tuple[pt.Tensor, list]:
    """
    :brief: normalize data to the interval [0, 1] using a min-max-normalization
    :param x: data which should be normalized
    :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
    """
    # x_i_normalized = (x_i - x_min) / (x_max - x_min)
    x_min_max = [pt.min(x), pt.max(x)]
    return pt.sub(x, x_min_max[0]) / (x_min_max[1] - x_min_max[0]), x_min_max


def denormalize_data(x: pt.Tensor, x_min_max: list) -> pt.Tensor:
    """
    :brief: reverse the normalization of the data
    :param x: normalized data
    :param x_min_max: min- and max-value used for normalizing the data
    :return: de-normalized data as tensor
    """
    # x = (x_max - x_min) * x_norm + x_min
    return (x_min_max[1] - x_min_max[0]) * x + x_min_max[0]


def dataloader_wrapper(settings: dict) -> dict:
    """
    :brief: load trajectory data, normalizes and splits the data into training-, validation- and testing data
    :param settings: setup defining paths, splitting rations etc.
    :return: dict containing all the trajectory data required for train, validate and testing the environment model
    """
    actions, states, c_l, c_d = load_trajectory_data(settings["load_path"])

    # resort data so that trajectory length per episode matches desired trajectory length for training
    print(f"data contains {actions.size()[-1]} trajectories with length of {actions.size()[0]} entries per trajectory")
    assert actions.size()[0] % settings["len_trajectory"] == 0, f"(trajectory length = {actions.size()[0]})" \
                                                             f"% (len_trajectory = {settings['len_trajectory']}) != 0 "
    actions = pt.concat(pt.split(actions, settings["len_trajectory"]), dim=1)
    c_l = pt.concat(pt.split(c_l, settings["len_trajectory"]), dim=1)
    c_d = pt.concat(pt.split(c_d, settings["len_trajectory"]), dim=1)
    states = pt.concat(pt.split(states, settings["len_trajectory"]), dim=2)
    all_data = {"n_probes": states.size()[1]}

    # normalize the data
    if settings["normalize"]:
        actions, all_data["min_max_actions"] = normalize_data(actions)
        c_l, all_data["min_max_cl"] = normalize_data(c_l)
        c_d, all_data["min_max_cd"] = normalize_data(c_d)
        states, all_data["min_max_states"] = normalize_data(states)

    # split dataset into training-, validation- and testdata
    all_data.update(split_data(states, actions, c_l, c_d, ratio=settings["ratio"]))

    # load value-, policy and MSE losses of PPO training
    all_data["network_data"] = pickle.load(open(settings["load_path"] + "training_history.pkl", "rb"))

    return all_data


def run_parameter_study(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                        batch_size: int = 50, epochs: int = 10000) -> Tuple[list[dict], list, list]:
    """
    :brief: initializes an environment model, trains and validates it based on the sampled data from the CFD
            environment; tests the best environment model based on test data
    :param settings: setup containing all paths and model setup
    :param trajectory_data: the loaded and split trajectories samples in the CFD environment
    :param n_neurons: number of neurons per layer in the environment model
    :param n_layers: number of hidden layers in the environment model
    :param epochs: number of epochs for training
    :param batch_size: batch size
    :return: predicted trajectories by the environment model, training and validation loss
    """
    # initialize environment network
    environment_model = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                n_outputs=trajectory_data["n_probes"] + 2, n_neurons=n_neurons, n_layers=n_layers)

    # train environment model
    train_loss, val_loss = train_model(environment_model, trajectory_data["states_train"],
                                       trajectory_data["actions_train"], trajectory_data["states_val"],
                                       trajectory_data["actions_val"], trajectory_data["cl_train"],
                                       trajectory_data["cd_train"], trajectory_data["cl_val"],
                                       trajectory_data["cd_val"], n_time_steps=settings["n_input_steps"],
                                       save_dir=settings["load_path"] + settings["model_dir"], save_name="bestModel",
                                       epochs=epochs, batch_size=batch_size)

    # test model: loop over all test data and predict the trajectories based on given initial state and actions
    environment_model.load_state_dict(pt.load(f"{settings['load_path'] + settings['model_dir']}/bestModel_train.pt"))
    prediction, shape = [], [settings["n_input_steps"], 1]
    for i in range(trajectory_data["actions_test"].size()[1]):
        model_input = pt.concat([trajectory_data["states_test"][:settings["n_input_steps"], :, i],
                                 (trajectory_data["cl_test"][:settings["n_input_steps"], i]).reshape(shape),
                                 (trajectory_data["cd_test"][:settings["n_input_steps"], i]).reshape(shape),
                                 (trajectory_data["actions_test"][:settings["n_input_steps"], i]).reshape(shape)],
                                dim=1)
        prediction.append(predict_trajectories(environment_model, model_input, trajectory_data["actions_test"][:, i],
                                               trajectory_data["n_probes"]))
    return prediction, train_loss, val_loss


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"/media/janis/Daten/Studienarbeit/drlfoam/examples/test_training3/",
        "path_to_probes": r"base/postProcessing/probes/0/",
        "model_dir": "Results_model/normalized_data/influenceModelArchitecture",
        "normalize": True,                                   # True: data will be normalized to interval of [1, 0]
        "n_input_steps": 2,                                  # initial time steps as input -> n_input_steps > 1
        "len_trajectory": 200,                               # trajectory length for training the environment model
        "ratio": (0.65, 0.3, 0.05),                          # splitting ratio for train-, validation and test data
        "n_neurons": 50,                                     # number of neurons per layer which should be tested
        "n_layers": 3                                        # number of hidden layers which should be tested
    }

    # load the sampled trajectories divided into training-, validation- and test data
    pt.Generator().manual_seed(0)                           # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    # initialize and train environment network, test best model
    pred_trajectory, train_loss, val_loss = run_parameter_study(setup, divided_data, n_neurons=setup["n_neurons"],
                                                                n_layers=setup["n_layers"])

    # create directory for plots
    if not os.path.exists(setup["load_path"] + setup["model_dir"] + "/plots"):
        os.mkdir(setup["load_path"] + setup["model_dir"] + "/plots")

    # plot training and validation loss
    plt.plot(range(len(train_loss)), train_loss, color="black", label="training loss")
    plt.plot(range(len(train_loss)), val_loss, color="red", label="validation loss")
    plt.xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    plt.ylabel("$MSE$ $loss$", usetex=True, fontsize=13)
    plt.yscale("log")
    plt.legend()
    plt.savefig(setup["load_path"] + setup["model_dir"] + "/plots/training_validation_loss_normalized.png", dpi=600)
    plt.show()

    # de-normalize the test data
    if setup["normalize"]:
        divided_data["cl_test"] = denormalize_data(divided_data["cl_test"], divided_data["min_max_cl"])
        divided_data["cd_test"] = denormalize_data(divided_data["cd_test"], divided_data["min_max_cd"])
        divided_data["states_test"] = denormalize_data(divided_data["states_test"], divided_data["min_max_states"])

    """
    # import coordinates of probes and plot their location
    probe_pos = import_probe_locations(setup["load_path"] + setup["path_to_probes"])
    plt.figure(num=1, figsize=(5, 5))
    plt.plot(probe_pos[:, 0], probe_pos[:, 1], linestyle="None", marker="o", color="black", label="probes")
    plt.xlabel("x-position", usetex=True)
    plt.ylabel("y-position", usetex=True)
    plt.legend()
    """

    # calculate the mean and std. deviation prediction error along the trajectories for all tested data
    error = {"error_cl": pt.zeros(divided_data["cl_test"].size()), "error_cd": pt.zeros(divided_data["cd_test"].size())}
    for i in range(divided_data["cl_test"].size()[1]):
        # reverse normalization of output data (= predicted trajectories)
        if setup["normalize"]:
            pred_trajectory[i]["cl"] = denormalize_data(pred_trajectory[i]["cl"], divided_data["min_max_cl"])
            pred_trajectory[i]["cd"] = denormalize_data(pred_trajectory[i]["cd"], divided_data["min_max_cd"])
            pred_trajectory[i]["states"] = denormalize_data(pred_trajectory[i]["states"], divided_data["min_max_states"])

        error["error_cl"][:, i] = pt.sub(pred_trajectory[i]["cl"], divided_data["cl_test"][:, i])
        error["error_cd"][:, i] = pt.sub(pred_trajectory[i]["cd"], divided_data["cd_test"][:, i])

    error["mean_cl"] = pt.mean(error["error_cl"], dim=1).detach().numpy()
    error["mean_cd"] = pt.mean(error["error_cd"], dim=1).detach().numpy()
    error["std_cl"] = pt.std(error["error_cl"], dim=1).detach().numpy()
    error["std_cd"] = pt.std(error["error_cd"], dim=1).detach().numpy()

    # plot mean and std. dev.
    plt.plot(range(setup["len_trajectory"]), error["mean_cl"], color="blue", label="$c_l$")
    plt.fill_between(range(setup["len_trajectory"]), error["mean_cl"] - error["std_cl"],
                     error["mean_cl"] + error["std_cl"], color="blue", alpha=0.3)
    plt.plot(range(setup["len_trajectory"]), error["mean_cd"], color="green", label="$c_d$")
    plt.fill_between(range(setup["len_trajectory"]), error["mean_cd"] - error["std_cd"],
                     error["mean_cd"] + error["std_cd"], color="green", alpha=0.3)
    plt.legend(loc="upper left", framealpha=1.0, fontsize=12, ncol=2)
    plt.xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    plt.ylabel("$total$ $prediction$ $error$", usetex=True, fontsize=13)
    plt.savefig(setup["load_path"] + setup["model_dir"] + "/plots/total_prediction_error_test_data.png", dpi=600)
    plt.show()

    # plot states of each probe for a random trajectory within the test data set and compare with model prediction
    trajectory_no = pt.randint(low=0, high=divided_data["actions_test"].size()[1], size=[1, 1]).item()
    fig1, ax1 = plt.subplots(nrows=divided_data["n_probes"], ncols=1, figsize=(9, 9), sharex="all", sharey="all")
    for i in range(divided_data["n_probes"]):
        ax1[i].plot(range(setup["len_trajectory"]), divided_data["states_test"][:, i, trajectory_no], color="black")
        ax1[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=40, usetex=True, fontsize=13)
        ax1[i].plot(range(setup["len_trajectory"]), pred_trajectory[trajectory_no]["states"][:, i].detach().numpy(),
                    color="red")
    fig1.subplots_adjust(hspace=0.75)
    ax1[-1].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    fig1.tight_layout()
    plt.savefig(setup["load_path"] + setup["model_dir"] + "/plots/real_trajectories_vs_prediction.png", dpi=600)
    plt.show()

    # compare real cl and cd values sampled in CFD environment with predicted ones along the trajectory
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(2):
        if i == 0:
            ax2[i].plot(range(setup["len_trajectory"]), divided_data["cl_test"][:, trajectory_no], color="black",
                        label="real")
            ax2[i].plot(range(setup["len_trajectory"]), pred_trajectory[trajectory_no]["cl"].detach().numpy(),
                        color="red",
                        label="prediction")
        else:
            ax2[i].plot(range(setup["len_trajectory"]), divided_data["cd_test"][:, trajectory_no], color="black")
            ax2[i].plot(range(setup["len_trajectory"]), pred_trajectory[trajectory_no]["cd"].detach().numpy(),
                        color="red")
    ax2[0].set_ylabel("$lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)
    ax2[0].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    ax2[1].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    ax2[1].set_ylabel("$drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)
    fig2.suptitle("coefficients - real vs. prediction", usetex=True, fontsize=16)
    fig2.tight_layout()
    fig2.legend(loc="upper right", framealpha=1.0, fontsize=12, ncol=2)
    fig2.subplots_adjust(wspace=0.25)
    plt.savefig(setup["load_path"] + setup["model_dir"] + "/plots/real_cl_cd_vs_prediction.png", dpi=600)
    plt.show()
