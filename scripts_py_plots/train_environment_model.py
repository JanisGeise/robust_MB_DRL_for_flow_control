"""
    brief:
        - implements and trains a fully connected NN-model for approximating the CFD environment using the sampled
          trajectories as trainings-, validation- and test data
        - tries to predict trajectories based on a given initial state and actions

        option 1:
            - using function 'train_test_env_model' trains- and tests only one environment model which is supposed to
              work for the whole data set independent of the episode
            - in order to use set parameter 'episode_depending_model' to 'False' within the 'setup' dict

        option 2:
            - using function 'train_test_env_model_episode_wise' trains- and tests models based wrt the episode
            - therefore:
                    - trajectories of 2 episodes (e.g. episode 1 and 2) are taken for training an environment model,
                      then it is used in order to predict trajectories of the next episode (e.g. episode 3)
                    - then a new model is trained based on the following two episodes (in this case episode 2 and 3)
                      and the trajectories of the next episode (e.g. episode 4) are predicted and so on
            - in order to use set parameter 'episode_depending_model' to 'True' within the 'setup' dict

    dependencies:
        - 'post_process_results_env_model.py' if the results should be post-processed and plotted

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import os
import pickle
import psutil
import torch as pt
import numpy as np
from glob import glob
from typing import Tuple, Union

from post_process_results_env_model import *


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
    creates feature-label pairs for in- / output of environment model

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
                action_val: pt.Tensor, cl_train: pt.Tensor, cd_train: pt.Tensor, cl_val: pt.Tensor, cd_val: pt.Tensor,
                epochs: int = 10000, lr: float = 0.0005, batch_size: int = 50, n_time_steps: int = 3,
                save_model: bool = True, save_name: str = "best", save_dir: str = "EnvModel",
                info: bool = False) -> Tuple[list, list]:
    """
    train environment model based on sampled trajectories

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
    :param info: print core temperature of processor every 250 epochs
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
            if info:
                print_core_temp()

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


def load_trajectory_data(path: str, preserve_episodes: bool = False, len_traj: int = 400) -> dict:
    """
    load observations_*.pkl files containing all the data generated during training and sort them into a dict

    :param path: path to directory containing the files
    :param preserve_episodes: either 'True' if the data should be sorted wrt the episodes, or 'False' if the order of
                              the episodes doesn't matter (in case only one model is trained for all the data)
    :param len_traj: length of the trajectories defined in the setup, the loaded trajectories are split wrt this length
    :return: actions, states, cl, cd as tensors within a dict where every column is a trajectory, also return number of
             workers used for sampling the trajectories. The structure is either:
             - all trajectories of each parameter in one tensor, independent of the episode. The resulting length
               of the trajectories corresponds to the length defined in the setup. This is the case if
               'preserve_episodes = False'

             - each parameter contains one tensor with the length of N_episodes. Each entry contains all the
               trajectories sampled in this episode (columns = N_trajectories, rows = length_trajectories).
               This is the case if 'preserve_episodes = True'
    """
    # load the 'observations_*.pkl' files containing the trajectories sampled in the CFD environment
    files = glob(path + "observations_*.pkl")
    observations = [pickle.load(open(file, "rb")) for file in files]
    actual_traj_length = len(observations[0][0]["actions"])

    # make sure there are no invalid settings defined
    assert actual_traj_length % len_traj == 0, f"(trajectory length = {actual_traj_length}) % (len_trajectory =" \
                                               f"{len_traj}) != 0 "
    assert actual_traj_length >= len_traj, f"imported trajectories can't be exended from {actual_traj_length} to" \
                                           f" {len_traj}!"

    data = {"n_workers": len(observations[0])}

    # if training is episode wise: sort the trajectories from all workers wrt the episode
    if preserve_episodes:
        factor = len_traj / actual_traj_length
        shape = (actual_traj_length, data["n_workers"])
        actions, cl, cd = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
        states = pt.zeros((shape[0], observations[0][0]["states"].size()[1], shape[1]))
        shape = (len(observations), int(actual_traj_length * factor), int(data["n_workers"] / factor))
        data["actions"], data["cl"], data["cd"] = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)
        data["states"] = pt.zeros((shape[0], shape[1], observations[0][0]["states"].size()[1], shape[2]))
        for episode in range(len(observations)):
            for worker in range(len(observations[episode])):
                actions[:, worker] = observations[episode][worker]["actions"]
                states[:, :, worker] = observations[episode][worker]["states"]
                cl[:, worker] = observations[episode][worker]["cl"]
                cd[:, worker] = observations[episode][worker]["cd"]
            data["actions"][episode, :, :] = pt.concat(pt.split(actions, len_traj), dim=1)
            data["states"][episode, :, :] = pt.concat(pt.split(states, len_traj), dim=2)
            data["cl"][episode, :, :] = pt.concat(pt.split(cl, len_traj), dim=1)
            data["cd"][episode, :, :] = pt.concat(pt.split(cd, len_traj), dim=1)

    # if only one model is trained using all available data, the order of the episodes doesn't matter
    else:
        shape = (actual_traj_length, len(observations) * len(observations[0]))
        n_probes, n_col = len(observations[0][0]["states"][0]), 0
        states = pt.zeros((shape[0], n_probes, shape[1]))
        actions, cl, cd = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape)

        for observation in range(len(observations)):
            for j in range(len(observations[observation])):
                actions[:, n_col] = observations[observation][j]["actions"]
                cl[:, n_col] = observations[observation][j]["cl"]
                cd[:, n_col] = observations[observation][j]["cd"]
                states[:, :, n_col] = observations[observation][j]["states"][:]
                n_col += 1

        data["actions"] = pt.concat(pt.split(actions, len_traj), dim=1)
        data["cl"] = pt.concat(pt.split(cl, len_traj), dim=1)
        data["cd"] = pt.concat(pt.split(cd, len_traj), dim=1)
        data["states"] = pt.concat(pt.split(states, len_traj), dim=2)

    return data


def split_data(states: pt.Tensor, actions: pt.Tensor, cl: pt.Tensor, cd: pt.Tensor, ratio: Tuple) -> dict:
    """
    split trajectories into train-, validation and test data

    :param states: sampled states in CFD environment
    :param actions: sampled actions in CFD environment
    :param cd: sampled cl-coefficients in CFD environment (at cylinder surface)
    :param cl: sampled cd-coefficients in CFD environment (at cylinder surface)
    :param ratio: ratio between training-, validation- and test data as tuple as (train, validation, test)
    :return: dictionary with splitted states and corresponding actions
    """
    data = {}
    # split dataset into training data, validation data and testdata
    if ratio[2] == 0:
        n_train, n_test = int(ratio[0] * actions.size()[1]), 0
        n_val = actions.size()[1] - n_train
    else:
        n_train, n_val = int(ratio[0] * actions.size()[1]), int(ratio[1] * actions.size()[1])
        n_test = actions.size()[1] - n_train - n_val

    # randomly select indices of trajectories
    samples = pt.ones(actions.shape[-1])
    idx_train = pt.multinomial(samples, n_train)
    idx_val = pt.multinomial(samples, n_val)

    # assign train-, validation and testing data based on chosen indices
    data["actions_train"], data["actions_val"] = actions[:, idx_train], actions[:, idx_val]
    data["states_train"], data["states_val"] = states[:, :, idx_train], states[:, :, idx_val]
    data["cl_train"], data["cd_train"] = cl[:, idx_train], cd[:, idx_train]
    data["cl_val"], data["cd_val"] = cl[:, idx_val], cd[:, idx_val]

    # if predictions are independent of episode: split test data, otherwise test data are the trajectories of the next
    # episode (take N episodes for training and try to predict episode N+1)
    if n_test != 0:
        idx_test = pt.multinomial(samples, n_test)
        data["cl_test"], data["cd_test"] = cl[:, idx_test], cd[:, idx_test]
        data["states_test"], data["actions_test"] = states[:, :, idx_test], actions[:, idx_test]

    return data


def predict_trajectories(model: pt.nn.Module, input_traj: pt.Tensor, actions: pt.Tensor, n_probes: int) -> dict:
    """
    using the environment model in order to predict the trajectory based on a given initial state and actions

    :param model: NN model of the environment
    :param input_traj: input states, cl, cd and actions for the first N time steps
    :param actions: actions taken in the CFD environment along the whole trajectory
    :param n_probes: number of probes used in the simulation
    :return: dictionary with the trajectories containing the states at the probe locations, cl, cd
    """
    # allocate tensor for predicted states, cl and cd for the whole trajectory and fill in the first N input states
    trajectory = pt.zeros([len(actions), input_traj.size()[1] - 1])
    trajectory[:input_traj.size()[0], :] = input_traj[:input_traj.size()[0], :-1]

    # loop over the trajectory
    for t in range(len(actions) - input_traj.size()[0]):
        # make prediction and append to trajectory tensor, then move input window by one time step
        feature = pt.flatten(pt.concat([trajectory[t:t + input_traj.size()[0], :],
                                        (actions[t:t + input_traj.size()[0]]).reshape([input_traj.size()[0], 1])],
                                       dim=1))
        trajectory[t + input_traj.size()[0], :] = model(feature).squeeze()

    # resort: divide output data into states, cl and cd
    output = {"states": trajectory[:, :n_probes], "cl": trajectory[:, -2], "cd": trajectory[:, -1]}
    return output


def normalize_data(x: pt.Tensor) -> Tuple[pt.Tensor, list]:
    """
    normalize data to the interval [0, 1] using a min-max-normalization

    :param x: data which should be normalized
    :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
    """
    # x_i_normalized = (x_i - x_min) / (x_max - x_min)
    x_min_max = [pt.min(x), pt.max(x)]
    return pt.sub(x, x_min_max[0]) / (x_min_max[1] - x_min_max[0]), x_min_max


def print_core_temp():
    """
    prints the current processor temperature of all available cores for monitoring

    :return: None
    """
    temps = psutil.sensors_temperatures()["coretemp"]
    print(f"{(4 * len(temps) + 1) * '-'}\n\tcore number\t|\ttemperature [deg]\t \n{(4 * len(temps) + 1) * '-'}")
    for i in range(len(temps)):
        print(f"\t\t{i}\t\t|\t{temps[i][1]}")


def dataloader_wrapper(settings: dict) -> dict:
    """
    load trajectory data, normalizes and splits the data into training-, validation- and testing data

    :param settings: setup defining paths, splitting rations etc.
    :return: dict containing all the trajectory data required for train, validate and testing the environment model
    """
    all_data = load_trajectory_data(settings["load_path"], settings["episode_depending_model"],
                                    settings["len_trajectory"])

    if not settings["episode_depending_model"]:
        print(f"data contains {all_data['actions'].size()[-1]} trajectories with length of"
              f" {all_data['actions'].size()[0]} entries per trajectory")
        all_data["n_probes"] = all_data["states"].size()[1]
    else:
        all_data["n_probes"] = all_data["states"][0].size()[1]

    # normalize the data
    if settings["normalize"]:
        all_data["actions"], all_data["min_max_actions"] = normalize_data(all_data["actions"])
        all_data["cl"], all_data["min_max_cl"] = normalize_data(all_data["cl"])
        all_data["cd"], all_data["min_max_cd"] = normalize_data(all_data["cd"])
        all_data["states"], all_data["min_max_states"] = normalize_data(all_data["states"])

    # split dataset into training-, validation- and test data if whole data set used for train only one (global) model
    if not settings["episode_depending_model"]:
        all_data.update(split_data(all_data["states"], all_data["actions"], all_data["cl"], all_data["cd"],
                                   ratio=settings["ratio"]))
        del all_data["actions"], all_data["states"], all_data["cl"], all_data["cd"]

    # load value-, policy and MSE losses of PPO training
    all_data["network_data"] = pickle.load(open(settings["load_path"] + "training_history.pkl", "rb"))

    return all_data


def train_test_env_model(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                         batch_size: int = 50, epochs: int = 10000) -> Tuple[list[dict], list, list]:
    """
    initializes an environment model, trains and validates it based on the sampled data from the CFD environment;
    tests the best environment model based on test data

    here: one model for the whole data set is trained and tested, independent of the episode

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
    train_mse, val_mse = train_model(environment_model, trajectory_data["states_train"],
                                     trajectory_data["actions_train"], trajectory_data["states_val"],
                                     trajectory_data["actions_val"], trajectory_data["cl_train"],
                                     trajectory_data["cd_train"], trajectory_data["cl_val"],
                                     trajectory_data["cd_val"], n_time_steps=settings["n_input_steps"],
                                     save_dir=settings["load_path"] + settings["model_dir"], save_name="bestModel",
                                     epochs=epochs, batch_size=batch_size, info=settings["print_temp"])

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
    return prediction, train_mse, val_mse


def train_test_env_model_episode_wise(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                                      batch_size: int = 50,
                                      epochs: int = 2500) -> Tuple[list[list[dict]], pt.Tensor, pt.Tensor]:
    """
    initializes an environment model, trains and validates it based on the sampled data from the CFD
    environment; tests the best environment model based on test data

    here: for each episode a new model is trained based on the trajectories of the previous two episodes

    :param settings: setup containing all paths and model setup
    :param trajectory_data: the loaded trajectories sampled in the CFD environment
    :param n_neurons: number of neurons per layer in the environment model
    :param n_layers: number of hidden layers in the environment model
    :param epochs: number of epochs for training
    :param batch_size: batch size
    :return: predicted trajectories by the environment model, training and validation loss
             the return values don't contain the first 2 episodes since there are no prediction available
    """
    n_episodes = trajectory_data["states"].size()[0]
    print(f"found {n_episodes} episodes in total")

    # the first two episodes are not predicted by any model, because training requires 2 episodes each
    train_mse, val_mse = [], []
    prediction, shape = [], [settings["n_input_steps"], 1]

    # for each episode init a new model, train it using 2 episodes, then test it using trajectories of the next episode
    for episode in range(1, n_episodes - 1):
        print(f"starting training of environment model for episode {episode + 2}")

        # initialize environment network for each new episode
        environment_model = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                    n_outputs=trajectory_data["n_probes"] + 2, n_neurons=n_neurons, n_layers=n_layers)

        # prepare the training-, validation and test data: 2 episodes for train- and validation, next episode as test
        states = pt.concat((trajectory_data["states"][episode - 1], trajectory_data["states"][episode]), dim=2)
        actions = pt.concat((trajectory_data["actions"][episode - 1], trajectory_data["actions"][episode]), dim=1)
        cl = pt.concat((trajectory_data["cl"][episode - 1], trajectory_data["cl"][episode]), dim=1)
        cd = pt.concat((trajectory_data["cd"][episode - 1], trajectory_data["cd"][episode]), dim=1)
        episode_data = split_data(states, actions, cl, cd, settings["ratio"])
        episode_data["states_test"] = trajectory_data["states"][episode + 1]

        # train environment model
        tmp_train_loss, tmp_val_loss = train_model(environment_model, episode_data["states_train"],
                                                   episode_data["actions_train"], episode_data["states_val"],
                                                   episode_data["actions_val"], episode_data["cl_train"],
                                                   episode_data["cd_train"], episode_data["cl_val"],
                                                   episode_data["cd_val"], n_time_steps=settings["n_input_steps"],
                                                   save_dir=settings["load_path"] + settings["model_dir"],
                                                   save_name=f"bestModel_episode{episode + 2}", epochs=epochs,
                                                   batch_size=batch_size, info=settings["print_temp"])

        # test model: loop over all test data and predict the trajectories based on given initial state and actions
        environment_model.load_state_dict(pt.load(f"{settings['load_path'] + settings['model_dir']}/"
                                                  f"bestModel_episode{episode + 2}_train.pt"))

        # loop over every trajectory within the current episode and try to predict the trajectories
        tmp_prediction = []
        for tra in range(trajectory_data["actions"][0].size()[1]):
            model_input = pt.concat([trajectory_data["states"][episode + 1][:settings["n_input_steps"], :, tra],
                                     (trajectory_data["cl"][episode + 1][:settings["n_input_steps"], tra]).reshape(
                                         shape),
                                     (trajectory_data["cd"][episode + 1][:settings["n_input_steps"], tra]).reshape(
                                         shape),
                                     (trajectory_data["actions"][episode + 1][:settings["n_input_steps"], tra]).reshape(
                                         shape)],
                                    dim=1)
            tmp_prediction.append(predict_trajectories(environment_model, model_input,
                                                       trajectory_data["actions"][episode + 1][:, tra],
                                                       trajectory_data["n_probes"]))

        prediction.append(tmp_prediction)
        train_mse.append(tmp_train_loss)
        val_mse.append(tmp_val_loss)

        # only keep the best model of each episode
        for file in glob(settings['load_path'] + settings['model_dir'] + "/*_epoch*.pt"):
            os.remove(file)

    return prediction, pt.tensor(train_mse).swapaxes(0, 1), pt.tensor(val_mse).swapaxes(0, 1)


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"/media/janis/Daten/Studienarbeit/drlfoam/examples/test_training3/",
        "path_to_probes": r"base/postProcessing/probes/0/",
        "model_dir": "Results/Results_model/one_model_for_each_episode/EnvironmentModel_2timesteps",
        "episode_depending_model": True,            # either one model for whole data set or new model for each episode
        "print_temp": False,                        # print core temperatur of processor as info
        "normalize": True,                          # True: data will be normalized to interval of [1, 0]
        "n_input_steps": 2,                         # initial time steps as input -> n_input_steps > 1
        "len_trajectory": 200,                      # trajectory length for training the environment model
        "ratio": (0.65, 0.35, 0.0),                 # splitting ratio for train-, validation and test data
        "epochs": 10000,                            # number of epochs to run for each model
        "n_neurons": 50,                            # number of neurons per layer which should be tested
        "n_layers": 3,                              # number of hidden layers which should be tested
        "n_seeds": 3                                # number of seed values over which should be averaged
    }
    if setup["episode_depending_model"]:
        assert setup["ratio"][2] == 0, "for episode depending model the test data ratio must be set to zero!"

    # TO_DO: implement loop for avg. over different seeds
    # load the sampled trajectories divided into training-, validation- and test data
    pt.Generator().manual_seed(0)                           # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    if setup["episode_depending_model"]:
        pred_trajectory, train_loss, val_loss = train_test_env_model_episode_wise(setup, divided_data,
                                                                                  n_neurons=setup["n_neurons"],
                                                                                  n_layers=setup["n_layers"],
                                                                                  epochs=setup["epochs"])

    else:
        # initialize and train environment network, test best model
        pred_trajectory, train_loss, val_loss = train_test_env_model(setup, divided_data, n_neurons=setup["n_neurons"],
                                                                     n_layers=setup["n_layers"], epochs=setup["epochs"])

    # create directory for plots and post-process the data
    if not os.path.exists(setup["load_path"] + setup["model_dir"] + "/plots"):
        os.mkdir(setup["load_path"] + setup["model_dir"] + "/plots")

    if setup["episode_depending_model"]:
        post_process_results_episode_wise_model(setup, pred_trajectory, divided_data, train_loss, val_loss)
    else:
        post_process_results_one_model(setup, pred_trajectory, divided_data, train_loss, val_loss)
