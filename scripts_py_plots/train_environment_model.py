"""
    brief:
        - implements and trains a fully connected NN-model for approximating the CFD environment using the sampled
          trajectories as trainings-, validation- and test data
        - tries to predict trajectories based on a given initial state and actions

        option 1a:
            - using function 'train_test_env_model' trains- and tests only one environment model which is supposed to
              work for the whole data set independent of the episode
            - in order to use set parameter 'episode_depending_model' to 'False' within the 'setup' dict

        option 2a:
            - using function 'train_test_env_model_episode_wise' trains- and tests models based wrt the episode
            - therefore:
                    - trajectories of 2 episodes (e.g. episode 1 and 2) are taken for training an environment model,
                      then it is used in order to predict trajectories of the next episode (e.g. episode 3)
                    - then a new model is trained based on the following two episodes (in this case episode 2 and 3)
                      and the trajectories of the next episode (e.g. episode 4) are predicted and so on
            - in order to use set parameter 'episode_depending_model' to 'True' within the 'setup' dict

        option 1b, and 2b:
            - same as option 1a and 2a but this time, two environment models are used
            - the first model is trained in order to predict the trajectories for cd
            - the second model is trained in order to predict the trajectories of the probes and cl

    dependencies:
        - 'utils.py'
        - 'post_process_results_env_model.py' if the results should be post-processed and plotted

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import os

from utils import *
from post_process_results_env_model import *


class FCModel(pt.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_neurons, activation: callable = pt.nn.functional.relu):
        """
        implements a fully-connected neural network

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
                               cl: pt.Tensor, cd: pt.Tensor, action: pt.Tensor, cd_model: bool = False,
                               two_models: bool = False) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    creates feature-label pairs for in- / output of environment model

    :param t_idx: index within the trajectory used as starting point for input states
    :param idx_trajectory: number of the trajectory within the whole training data set
    :param n_time_steps: number of time steps used for input
    :param trajectory: the trajectory used as feature / label
    :param cl: corresponding cl coefficient to the in- and output states
    :param cd: corresponding cd coefficient to the in- and output states
    :param action: actions taken in the input states
    :param cd_model: flag if this model is for predicting cd only
    :param two_models: flag if one environment or two environment models are used in order to predict cd, cl, probes
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

    # labels contain next step to predict by the model, if only one model is used then label = cd, cl, probes
    if not two_models:
        label = pt.concat([trajectory[t_idx + n_time_steps, :].squeeze(), cl[t_idx + n_time_steps, idx_trajectory],
                           cd[t_idx + n_time_steps, idx_trajectory]], dim=0)

    # else: label is either only cd, or cl and probes depending on which model
    else:
        if cd_model:
            label = cd[t_idx + n_time_steps, idx_trajectory]

        else:
            label = pt.concat([trajectory[t_idx + n_time_steps, :].squeeze(), cl[t_idx + n_time_steps, idx_trajectory]],
                              dim=0)

    return pt.flatten(feature), pt.flatten(label)


def train_model(model: pt.nn.Module, state_train: pt.Tensor, action_train: pt.Tensor, state_val: pt.Tensor,
                action_val: pt.Tensor, cl_train: pt.Tensor, cd_train: pt.Tensor, cl_val: pt.Tensor, cd_val: pt.Tensor,
                epochs: int = 10000, lr: float = 0.0005, batch_size: int = 50, n_time_steps: int = 3,
                save_model: bool = True, save_name: str = "best", save_dir: str = "EnvModel",
                info: bool = False, cd_model: bool = False, two_models: bool = False) -> Tuple[list, list]:
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
    :param cd_model: flag if this model is for predicting cd only
    :param two_models: flag if one environment or two environment models are used in order to predict cd, cl, probes
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
                                                        cl_train, cd_train, action_train, cd_model, two_models)

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
                                                            cl_val, cd_val, action_val, cd_model, two_models)
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


def train_test_env_model(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                         batch_size: int = 50, epochs: int = 10000) -> Tuple[list[dict], list, list]:
    """
    initializes an environment model, trains and validates it based on the sampled data from the CFD environment;
    tests the best environment model based on test data (option 1a)

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
    environment_model.load_state_dict(pt.load("".join([settings["load_path"], settings["model_dir"],
                                                       f"/bestModel_train.pt"])))
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
    environment; tests the best environment model based on test data (option 2a)

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
        environment_model.load_state_dict(pt.load("".join([settings["load_path"], settings["model_dir"],
                                                           f"/bestModel_episode{episode + 2}_train.pt"])))

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
        for file in glob("".join([settings['load_path'], settings['model_dir'], "/*_epoch*.pt"])):
            os.remove(file)

    return prediction, pt.tensor(train_mse).swapaxes(0, 1), pt.tensor(val_mse).swapaxes(0, 1)


def env_model_2models(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                      batch_size: int = 50, epochs: int = 10000) -> Tuple[list[dict], list, list]:
    """
    initializes an environment model for cl and probes as well as a 2nd model for cd, trains and validates them based on
    the sampled data from the CFD environment; tests the best environment models based on test data (option 1b)

    here: one model for the whole data set is trained and tested respectively, independent of the episode

    :param settings: setup containing all paths and model setup
    :param trajectory_data: the loaded and split trajectories samples in the CFD environment
    :param n_neurons: number of neurons per layer in the environment model
    :param n_layers: number of hidden layers in the environment model
    :param epochs: number of epochs for training
    :param batch_size: batch size
    :return: predicted trajectories by the environment model, training and validation loss
    """
    if not os.path.exists(settings["load_path"] + settings["model_dir"]):
        os.mkdir(settings["load_path"] + settings["model_dir"])

    # initialize environment networks
    environment_model_1 = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                  n_outputs=trajectory_data["n_probes"] + 1, n_neurons=n_neurons, n_layers=n_layers)
    environment_model_cd = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                   n_outputs=1, n_neurons=settings["n_neurons_cd"], n_layers=settings["n_layers_cd"])

    print("starting training for probes & cl environment model")
    # train environment model
    train_mse, val_mse = train_model(environment_model_1, trajectory_data["states_train"],
                                     trajectory_data["actions_train"], trajectory_data["states_val"],
                                     trajectory_data["actions_val"], trajectory_data["cl_train"],
                                     trajectory_data["cd_train"], trajectory_data["cl_val"],
                                     trajectory_data["cd_val"], n_time_steps=settings["n_input_steps"],
                                     save_dir="".join([settings["load_path"], settings["model_dir"], "/env_model_1/"]),
                                     save_name="bestModel", epochs=epochs, batch_size=batch_size,
                                     info=settings["print_temp"], cd_model=False,
                                     two_models=settings["two_env_models"])

    print("starting training for cd environment model")
    train_mse_cd, val_mse_cd = train_model(environment_model_cd, trajectory_data["states_train"],
                                           trajectory_data["actions_train"], trajectory_data["states_val"],
                                           trajectory_data["actions_val"], trajectory_data["cl_train"],
                                           trajectory_data["cd_train"], trajectory_data["cl_val"],
                                           trajectory_data["cd_val"], n_time_steps=settings["n_input_steps"],
                                           save_dir="".join([settings["load_path"], settings["model_dir"], "/cd_model/"]),
                                           save_name="bestModel", epochs=settings["epochs_cd"],
                                           batch_size=10, info=settings["print_temp"], cd_model=True,
                                           two_models=settings["two_env_models"])

    # test model: loop over all test data and predict the trajectories based on given initial state and actions
    environment_model_1.load_state_dict(pt.load("".join([settings["load_path"], settings["model_dir"],
                                                         f"/env_model_1/bestModel_train.pt"])))
    environment_model_cd.load_state_dict(pt.load("".join([settings["load_path"], settings["model_dir"],
                                                         f"/cd_model/bestModel_train.pt"])))
    prediction, shape = [], [settings["n_input_steps"], 1]
    for i in range(trajectory_data["actions_test"].size()[1]):
        model_input = pt.concat([trajectory_data["states_test"][:settings["n_input_steps"], :, i],
                                 (trajectory_data["cl_test"][:settings["n_input_steps"], i]).reshape(shape),
                                 (trajectory_data["cd_test"][:settings["n_input_steps"], i]).reshape(shape),
                                 (trajectory_data["actions_test"][:settings["n_input_steps"], i]).reshape(shape)],
                                dim=1)
        env1_model_pred = predict_trajectories_2models(environment_model_1, model_input,
                                                       trajectory_data["states_test"][:, :, i],
                                                       trajectory_data["cd_test"][:, i],
                                                       trajectory_data["cl_test"][:, i],
                                                       trajectory_data["actions_test"][:, i],
                                                       trajectory_data["n_probes"],
                                                       cd_model=False)
        env1_model_pred["cd"] = predict_trajectories_2models(environment_model_cd, model_input,
                                                             trajectory_data["states_test"][:, :, i],
                                                             trajectory_data["cd_test"][:, i],
                                                             trajectory_data["cl_test"][:, i],
                                                             trajectory_data["actions_test"][:, i],
                                                             trajectory_data["n_probes"],
                                                             cd_model=True)
        prediction.append(env1_model_pred)

    return prediction, [train_mse, train_mse_cd], [val_mse, val_mse_cd]


def predict_trajectories_2models(model: pt.nn.Module, input_traj: pt.Tensor, states: pt.Tensor, cd: pt.Tensor,
                                 cl: pt.Tensor, actions: pt.Tensor, n_probes: int,
                                 cd_model: bool = False) -> dict or pt.Tensor:
    """
    using the two environment models in order to predict the trajectory based on a given initial state and actions
    (option 1b & 2b)

    :param model: NN model of the environment
    :param input_traj: input states, cl, cd and actions for the first N time steps
    :param actions: actions taken in the CFD environment along the whole trajectory
    :param n_probes: number of probes used in the simulation
    :param cl: trajectories of cl in the CFD environment
    :param cd: trajectories of cd in the CFD environment
    :param states: trajectories of the states at probe locations in the CFD environment
    :param cd_model: flag if this model is for predicting cd only
    :return: dictionary with the trajectories containing the states at the probe locations, cl, cd
    """
    # allocate tensor for predicted states, cl and cd for the whole trajectory and fill in the first N input states
    if cd_model:
        trajectory = pt.zeros(len(actions))
        trajectory[:input_traj.size()[0]] = cd[:input_traj.size()[0]]
    else:
        trajectory = pt.zeros([len(actions), input_traj.size()[1] - 2])
        trajectory[:input_traj.size()[0], :] = input_traj[:input_traj.size()[0], :-2]

    # loop over the trajectory
    for t in range(len(actions) - input_traj.size()[0]):
        if cd_model:
            # make prediction and append to trajectory tensor, then move input window by one time step
            feature = pt.flatten(pt.concat([states[t:t + input_traj.size()[0], :],
                                            (cl[t:t + input_traj.size()[0]]).reshape([input_traj.size()[0], 1]),
                                            trajectory[t:t + input_traj.size()[0]].reshape([input_traj.size()[0], 1]),
                                            (actions[t:t + input_traj.size()[0]]).reshape([input_traj.size()[0], 1])],
                                           dim=1))
            trajectory[t + input_traj.size()[0]] = model(feature).squeeze()
        else:
            # make prediction and append to trajectory tensor, then move input window by one time step
            feature = pt.flatten(pt.concat([trajectory[t:t + input_traj.size()[0], :],
                                            (cd[t:t + input_traj.size()[0]]).reshape([input_traj.size()[0], 1]),
                                            (actions[t:t + input_traj.size()[0]]).reshape([input_traj.size()[0], 1])],
                                           dim=1))
            trajectory[t + input_traj.size()[0], :] = model(feature).squeeze()

    # resort: divide output data into states and cl or just return the cd trajectory depending on the model
    if cd_model:
        return trajectory
    else:
        output = {"states": trajectory[:, :n_probes], "cl": trajectory[:, -1]}

        return output


def env_model_episode_wise_2models(settings: dict, trajectory_data: dict, n_neurons: int = 32, n_layers: int = 5,
                                   batch_size: int = 50, epochs: int = 2500) -> Tuple[list[list[dict]], list, list]:
    """
    initializes two environment models, trains and validates them based on the sampled data from the CFD
    environment; tests the best environment models based on test data

    here: for each episode two new models are trained based on the trajectories of the previous two episodes

    :param settings: setup containing all paths and model setup
    :param trajectory_data: the loaded trajectories sampled in the CFD environment
    :param n_neurons: number of neurons per layer in the environment model
    :param n_layers: number of hidden layers in the environment model
    :param epochs: number of epochs for training
    :param batch_size: batch size
    :return: predicted trajectories by the environment model, training and validation loss
             the return values don't contain the first 2 episodes since there are no prediction available
    """
    if not os.path.exists(settings["load_path"] + settings["model_dir"]):
        os.mkdir(settings["load_path"] + settings["model_dir"])

    n_episodes = trajectory_data["states"].size()[0]
    print(f"found {n_episodes} episodes in total")

    # the first two episodes are not predicted by any model, because training requires 2 episodes each
    # train_mse, val_mse = pt.zeros((epochs, n_episodes)), pt.zeros((epochs, n_episodes))
    train_mse, val_mse, train_mse_cd, val_mse_cd = [], [], [], []
    prediction, shape = [], [settings["n_input_steps"], 1]

    # for each episode init a new model, train it using 2 episodes, then test it using trajectories of the next episode
    for episode in range(1, n_episodes - 1):
        print(f"starting training of environment model for episode {episode + 2}")

        # initialize environment networks
        environment_model_1 = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                      n_outputs=trajectory_data["n_probes"] + 1, n_neurons=n_neurons, n_layers=n_layers)
        environment_model_cd = FCModel(n_inputs=settings["n_input_steps"] * (trajectory_data["n_probes"] + 3),
                                       n_outputs=1, n_neurons=settings["n_neurons_cd"],
                                       n_layers=settings["n_layers_cd"])

        # prepare the training-, validation and test data: 2 episodes for train- and validation, next episode as test
        states = pt.concat((trajectory_data["states"][episode - 1], trajectory_data["states"][episode]), dim=2)
        actions = pt.concat((trajectory_data["actions"][episode - 1], trajectory_data["actions"][episode]), dim=1)
        cl = pt.concat((trajectory_data["cl"][episode - 1], trajectory_data["cl"][episode]), dim=1)
        cd = pt.concat((trajectory_data["cd"][episode - 1], trajectory_data["cd"][episode]), dim=1)
        episode_data = split_data(states, actions, cl, cd, settings["ratio"])
        episode_data["states_test"] = trajectory_data["states"][episode + 1]

        # train environment model
        tmp_train_loss, tmp_val_loss = train_model(environment_model_1, episode_data["states_train"],
                                                   episode_data["actions_train"], episode_data["states_val"],
                                                   episode_data["actions_val"], episode_data["cl_train"],
                                                   episode_data["cd_train"], episode_data["cl_val"],
                                                   episode_data["cd_val"],
                                                   n_time_steps=settings["n_input_steps"],
                                                   save_dir="".join([settings["load_path"], settings["model_dir"],
                                                                     "/env_model_1/"]),
                                                   save_name=f"bestModel_episode{episode + 2}", epochs=epochs,
                                                   batch_size=batch_size, info=settings["print_temp"],
                                                   cd_model=False, two_models=settings["two_env_models"])

        print("starting training for cd model")
        tmp_train_mse_cd, tmp_val_mse_cd = train_model(environment_model_cd, episode_data["states_train"],
                                                       episode_data["actions_train"],
                                                       episode_data["states_val"],
                                                       episode_data["actions_val"], episode_data["cl_train"],
                                                       episode_data["cd_train"], episode_data["cl_val"],
                                                       episode_data["cd_val"],
                                                       n_time_steps=settings["n_input_steps"],
                                                       save_dir="".join([settings["load_path"], settings["model_dir"],
                                                                         "/cd_model/"]),
                                                       save_name=f"bestModel_episode{episode + 2}",
                                                       epochs=settings["epochs_cd"], batch_size=10,
                                                       info=settings["print_temp"], cd_model=True,
                                                       two_models=settings["two_env_models"])

        # test model: loop over all test data and predict the trajectories based on given initial state and actions
        environment_model_1.load_state_dict(pt.load(f"{settings['load_path'] + settings['model_dir']}"
                                                    f"/env_model_1/bestModel_episode{episode + 2}_train.pt"))
        environment_model_cd.load_state_dict(pt.load(f"{settings['load_path'] + settings['model_dir']}"
                                                     f"/cd_model/bestModel_episode{episode + 2}_train.pt"))

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
            env1_model_pred = predict_trajectories_2models(environment_model_1, model_input,
                                                           trajectory_data["states"][episode + 1][:, :, tra],
                                                           trajectory_data["cd"][episode + 1][:, tra],
                                                           trajectory_data["cl"][episode + 1][:, tra],
                                                           trajectory_data["actions"][episode + 1][:, tra],
                                                           trajectory_data["n_probes"],
                                                           cd_model=False)
            env1_model_pred["cd"] = predict_trajectories_2models(environment_model_cd, model_input,
                                                                 trajectory_data["states"][episode + 1][:, :, tra],
                                                                 trajectory_data["cd"][episode + 1][:, tra],
                                                                 trajectory_data["cl"][episode + 1][:, tra],
                                                                 trajectory_data["actions"][episode + 1][:, tra],
                                                                 trajectory_data["n_probes"],
                                                                 cd_model=True)
            tmp_prediction.append(env1_model_pred)

        prediction.append(tmp_prediction)
        train_mse.append(tmp_train_loss)
        val_mse.append(tmp_val_loss)
        train_mse_cd.append(tmp_train_mse_cd)
        val_mse_cd.append(tmp_val_mse_cd)

        # only keep the best model of each episode
        for file in glob("".join([settings["load_path"], settings["model_dir"], "/**/*_epoch*.pt"]), recursive=True):
            os.remove(file)

    return prediction, [pt.tensor(train_mse).swapaxes(0, 1), pt.tensor(train_mse_cd).swapaxes(0, 1)], \
                       [pt.tensor(val_mse).swapaxes(0, 1), pt.tensor(val_mse_cd).swapaxes(0, 1)]


if __name__ == "__main__":
    # Setup
    setup = {
        "load_path": r"/media/janis/Daten/Studienarbeit/drlfoam/examples/test_training3/",
        "path_to_probes": r"base/postProcessing/probes/0/",
        "model_dir": "Results_model/TEST",
        "episode_depending_model": True,    # either one model for whole data set or new model for each episode
        "two_env_models": True,             # 'True': one model only for predicting cd, another for probes and cl
        "print_temp": False,                # print core temperatur of processor as info
        "normalize": True,                  # True: data will be normalized to interval of [1, 0]
        "n_input_steps": 3,                 # initial time steps as input -> n_input_steps > 1
        "len_trajectory": 200,              # trajectory length for training the environment model
        "ratio": (0.65, 0.35, 0.0),         # splitting ratio for train-, validation and test data
        "epochs": 10000,                    # number of epochs to run for the environment model
        "n_neurons": 50,                    # number of neurons per layer for the environment model
        "n_layers": 3,                      # number of hidden layers for the environment model
        "n_neurons_cd": 50,                 # number of neurons per layer for the env model for cd (if option is set)
        "n_layers_cd": 5,                   # number of hidden layers for the env model for cd (if option is set)
        "epochs_cd": 10000,                 # number of epochs to run for the env model for cd (if option is set)
    }
    if setup["episode_depending_model"]:
        assert setup["ratio"][2] == 0, "for episode depending model the test data ratio must be set to zero!"

    # load the sampled trajectories divided into training-, validation- and test data
    pt.Generator().manual_seed(0)                           # ensure reproducibility
    divided_data = dataloader_wrapper(settings=setup)

    # depending on which option are chosen train and test environment model(s)
    if setup["two_env_models"]:
        if setup["episode_depending_model"]:
            pred_trajectory, train_loss, val_loss = env_model_episode_wise_2models(setup, divided_data,
                                                                                   n_neurons=setup["n_neurons"],
                                                                                   n_layers=setup["n_layers"],
                                                                                   epochs=setup["epochs"])

        else:
            pred_trajectory, train_loss, val_loss = env_model_2models(setup, divided_data,
                                                                      n_neurons=setup["n_neurons"],
                                                                      n_layers=setup["n_layers"],
                                                                      epochs=setup["epochs"])

    else:
        if setup["episode_depending_model"]:
            pred_trajectory, train_loss, val_loss = train_test_env_model_episode_wise(setup, divided_data,
                                                                                      n_neurons=setup["n_neurons"],
                                                                                      n_layers=setup["n_layers"],
                                                                                      epochs=setup["epochs"])

        else:
            pred_trajectory, train_loss, val_loss = train_test_env_model(setup, divided_data,
                                                                         n_neurons=setup["n_neurons"],
                                                                         n_layers=setup["n_layers"],
                                                                         epochs=setup["epochs"])

    # create directory for plots and post-process the data
    if not os.path.exists("".join([setup["load_path"], setup["model_dir"], "/plots"])):
        os.mkdir("".join([setup["load_path"], setup["model_dir"], "/plots"]))

    if setup["two_env_models"]:
        if setup["episode_depending_model"]:
            # plot train- and validation loss of cd environment model
            plot_train_validation_loss(path=setup["load_path"] + setup["model_dir"],
                                       mse_train=pt.mean(train_loss[1], dim=0),
                                       mse_val=pt.mean(val_loss[1], dim=0), std_dev_train=pt.std(train_loss[1], dim=0),
                                       std_dev_val=pt.std(val_loss[1], dim=0), cd_model=True, episode_wise=True)

            plot_train_validation_loss(path=setup["load_path"] + setup["model_dir"],
                                       mse_train=pt.mean(train_loss[1], dim=1),
                                       mse_val=pt.mean(val_loss[1], dim=1), std_dev_train=pt.std(train_loss[1], dim=1),
                                       std_dev_val=pt.std(val_loss[1], dim=1), cd_model=True, episode_wise=False)

            # post-process remaining data
            post_process_results_episode_wise_model(setup, pred_trajectory, divided_data, train_loss[0], val_loss[0])

        else:
            # plot train- and validation loss of cd environment model
            plot_train_validation_loss(path=setup["load_path"] + setup["model_dir"], mse_train=train_loss[1],
                                       mse_val=val_loss[1], cd_model=True)

            # post-process remaining data
            post_process_results_one_model(setup, pred_trajectory, divided_data, train_loss[0], val_loss[0])

    else:
        if setup["episode_depending_model"]:
            post_process_results_episode_wise_model(setup, pred_trajectory, divided_data, train_loss, val_loss)
        else:
            post_process_results_one_model(setup, pred_trajectory, divided_data, train_loss, val_loss)
