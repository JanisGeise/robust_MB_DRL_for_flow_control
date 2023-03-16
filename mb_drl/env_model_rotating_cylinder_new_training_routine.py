"""
    This script is implements the model-based part for the PPO-training routine. It contains the environment model class
    as well as functions for loading and sorting the trajectories, training the model ensembles and generating the
    model-based trajectories for the PPO-training.
"""
import torch as pt
from os import mkdir
from typing import Tuple, List
from os.path import join, exists
from torch.utils.data import DataLoader, TensorDataset, random_split

from drlfoam.agent import PPOAgent
from drlfoam.agent.agent import compute_gae
from drlfoam.constants import EPS_SP


class SetupEnvironmentModel:
    def __init__(self, n_models: int = 5, n_input_time_steps: int = 30, path: str = ""):
        self.path = path
        self.n_models = n_models
        self.t_input = n_input_time_steps
        self.obs_cfd = []
        self.len_traj = 200
        self.last_cfd = 0
        self.policy_loss = []
        self.threshold = 0.5

    def determine_switching(self, current_episode: int):
        """
        check if the environment models are still improving the policy or if it should be switched back to CFD in order
        to update the environment models

        threshold: amount of model which have to improve the policy in order to continue training with models

        :param current_episode: current episode
        :return: bool if training should be switch back to CFD in order to update the environment models
        """
        # in the 1st two MB episodes of training, policy loss has no or just one entry, so diff can't be computed
        if len(self.policy_loss) < 2:
            switch = 0
        else:
            # mask difference of policy loss for each model -> 1 if policy improved, 0 if not
            diff = ((pt.tensor(self.policy_loss[-2]) - pt.tensor(self.policy_loss[-1])) > 0.0).int()

            # if policy for less than x% of the models improves, then switch to CFD -> 0 = no switching, 1 = switch
            switch = [0 if sum(diff) / len(diff) >= self.threshold else 1][0]

        # we need 2 subsequent MB-episodes after each CFD episode to determine if policy improves in MB-training
        if (current_episode - self.last_cfd < 2) or switch == 0:
            return False
        else:
            return True

    def append_cfd_obs(self, e):
        self.obs_cfd.append(join(self.path, f"observations_{e}.pt"))

    def save_losses(self, episode, loss):
        # save train- and validation losses of the environment models
        if self.n_models == 1:
            losses = {"train_loss_cl_p": loss[0][0], "train_loss_cd": loss[0][1], "val_loss_cl_p": loss[1][0],
                      "val_loss_cd": loss[1][1]}
            self.save(episode, losses, name="env_model_loss")
        else:
            losses = {"train_loss_cl_p": [l[0][0] for l in loss], "train_loss_cd": [l[0][1] for l in loss],
                      "val_loss_cl_p": [l[1][0] for l in loss], "val_loss_cd": [l[1][1] for l in loss]}
            self.save(episode, losses, name="env_model_loss")

    def save(self, episode, data, name: str = "observations"):
        pt.save(data, join(self.path, f"{name}_{episode}.pt"))

    def reset(self, episode):
        self.policy_loss = []
        self.last_cfd = episode


class EnvironmentModel(pt.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int, n_neurons: int,
                 activation: callable = pt.nn.functional.leaky_relu):
        """
        implements a fully-connected neural network

        :param n_inputs: N probes * N time steps + (1 action + 1 cl + 1 cd) * N time steps
        :param n_outputs: N probes
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: none
        """
        super(EnvironmentModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.layers = pt.nn.ModuleList()

        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
                self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)


def train_model(model: pt.nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader, epochs: int = 2500,
                lr: float = 0.01, stop: float = -1e-7, save_name: str = "bestModel",
                save_dir: str = "env_model") -> Tuple[list, list]:
    """
    train environment model based on sampled trajectories

    :param model: environment model
    :param dataloader_train: dataloader for training
    :param dataloader_val: dataloader for validation
    :param epochs: number of epochs for training
    :param lr: learning rate
    :param stop: if avg. gradient of validation loss reaches this value, the training is aborted
    :param save_model: option to save best model, default is True
    :param save_dir: path to directory where models should be saved
    :param save_name: name of the model saved, default is number of epoch
    :return: training and validation loss as list
    """
    if not exists(save_dir):
        mkdir(save_dir)

    # optimizer settings
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, min_lr=1.0e-4)

    # lists for storing losses
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    training_loss, validation_loss = [], []

    for epoch in range(1, epochs + 1):
        t_loss_tmp, v_loss_tmp = [], []

        # training loop
        model.train()
        for feature, label in dataloader_train:
            optimizer.zero_grad()
            prediction = model(feature).squeeze()
            loss_train = criterion(prediction, label.squeeze())
            loss_train.backward()
            optimizer.step()
            t_loss_tmp.append(loss_train.item())
        training_loss.append(pt.mean(pt.tensor(t_loss_tmp)).to("cpu"))

        # validation loop
        with pt.no_grad():
            for feature, label in dataloader_val:
                prediction = model(feature).squeeze()
                loss_val = criterion(prediction, label.squeeze())
                v_loss_tmp.append(pt.mean(loss_val).item())
        validation_loss.append(pt.mean(pt.tensor(v_loss_tmp)).to("cpu"))

        scheduler.step(metrics=validation_loss[-1])

        if validation_loss[-1] < best_val_loss:
            pt.save(model.state_dict(), join(save_dir, f"{save_name}_val.pt"))
            best_val_loss = validation_loss[-1]

        # print some info after every 100 epochs
        if epoch % 100 == 0:
            print(f"epoch {epoch}, avg. training loss = {round(pt.mean(pt.tensor(training_loss[-50:])).item(), 8)}, "
                  f"avg. validation loss = {round(pt.mean(pt.tensor(validation_loss[-50:])).item(), 8)}")

        # check every 50 epochs if model performs well on validation data or validation loss converges
        if epoch % 50 == 0 and epoch >= 150:
            avg_grad_val_loss = (pt.mean(pt.tensor(validation_loss[-5:-1])) -
                                 pt.mean(pt.tensor(validation_loss[-52:-48]))) / 48

            # since loss decreases the gradient is negative, so if it converges or starts increasing, then stop training
            if validation_loss[-1] <= 1e-6 or avg_grad_val_loss >= stop:
                break

    return training_loss, validation_loss


def normalize_data(x: pt.Tensor, x_min_max: list = None) -> Tuple[pt.Tensor, list]:
    """
    normalize data to the interval [0, 1] using a min-max-normalization

    :param x: data which should be normalized
    :param x_min_max: list containing the min-max-values for normalization (optional)
    :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
    """
    # x_i_normalized = (x_i - x_min) / (x_max - x_min)
    if x_min_max is None:
        x_min_max = [pt.min(x), pt.max(x)]
    return pt.sub(x, x_min_max[0]) / (x_min_max[1] - x_min_max[0]), x_min_max


def denormalize_data(x: pt.Tensor, x_min_max: list) -> pt.Tensor:
    """
    reverse the normalization of the data

    :param x: normalized data
    :param x_min_max: min- and max-value used for normalizing the data
    :return: de-normalized data as tensor
    """
    # x = (x_max - x_min) * x_norm + x_min
    return (x_min_max[1] - x_min_max[0]) * x + x_min_max[0]


def load_trajectory_data(files: list, len_traj: int, n_probes: int):
    """
    load the trajectory data from the observations_*.pt files

    :param files: list containing the file names of the last two episodes run in CFD environment
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param n_probes: number of probes placed in the flow field
    :return: cl, cd, actions, states, alpha, beta
    """
    # in new version of drlfoam the observations are in stored in '.pt' files
    observations = [pt.load(open(file, "rb")) for file in files]

    # sort the trajectories from all workers, for training the models, it doesn't matter from which episodes the data is
    shape, n_col = (len_traj, len(observations) * len(observations[0])), 0
    states = pt.zeros((shape[0], n_probes, shape[1]))
    actions, cl, cd, alpha, beta, rewards = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), pt.zeros(shape), \
                                            pt.zeros(shape), pt.zeros(shape)

    for observation in range(len(observations)):
        for j in range(len(observations[observation])):
            # omit failed or partly converged trajectories
            if not bool(observations[observation][j]) or observations[observation][j]["actions"].size()[0] < len_traj:
                pass
            # for some reason sometimes the trajectories are 1 entry too long, in that case ignore the last value
            else:
                actions[:, n_col] = observations[observation][j]["actions"][:len_traj]
                cl[:, n_col] = observations[observation][j]["cl"][:len_traj]
                cd[:, n_col] = observations[observation][j]["cd"][:len_traj]
                alpha[:, n_col] = observations[observation][j]["alpha"][:len_traj]
                beta[:, n_col] = observations[observation][j]["beta"][:len_traj]
                rewards[:, n_col] = observations[observation][j]["rewards"][:len_traj]
                states[:, :, n_col] = observations[observation][j]["states"][:len_traj][:]
            n_col += 1

    return cl, cd, actions, states, alpha, beta, rewards


def check_cfd_data(files: list, len_traj: int, n_probes: int, buffer_size: int = 10, n_e_cfd: int = 0) -> dict:
    """
    load the trajectory data, split the trajectories into training, validation- and test data (for sampling initial
    states), normalize all the data to [0, 1]

    :param files: list containing the file names of the last two episodes run in CFD environment
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param n_probes: number of probes placed in the flow field
    :param buffer_size: current buffer size
    :param n_e_cfd: number of currently available episodes run in CFD
    :return: dict containing the loaded, sorted and normalized data as well as the data for training- and validation
    """
    data, idx_train, idx_val = {}, 0, 0
    cl, cd, actions, states, alpha, beta, rewards = load_trajectory_data(files[-2:], len_traj, n_probes)

    # delete the allocated columns of the failed trajectories (since they would alter the mean values)
    ok = cl.abs().sum(dim=0).bool()
    ok_states = states.abs().sum(dim=0).sum(dim=0).bool()

    # if buffer of current CFD episode is empty, then only one CFD episode can be used for training which is likely to
    # crash. In this case, load trajectories from 3 CFD episodes ago (e-2 wouldn't work, since e-2 and e-1 where used to
    # train the last model ensemble)
    if len([i.item() for i in ok if i]) <= buffer_size and n_e_cfd > 3:
        # determine current episode and then take the 'observations_*.pkl' from 3 CFD episodes ago
        cl_tmp, cd_tmp, actions_tmp, states_tmp, alpha_tmp, beta_tmp, rewards_tmp = load_trajectory_data([files[-4]],
                                                                                                         len_traj,
                                                                                                         n_probes)

        # merge into existing data
        cl, cd, actions, states = pt.concat([cl, cl_tmp], dim=1), pt.concat([cd, cd_tmp], dim=1), \
                                  pt.concat([actions, actions_tmp], dim=1), pt.concat([states, states_tmp], dim=2)
        alpha, beta, rewards = pt.concat([alpha, alpha_tmp], dim=1), pt.concat([beta, beta_tmp], dim=1), \
                               pt.concat([rewards, rewards_tmp], dim=1)

        # and finally update the masks to remove non-converged trajectories
        ok = cl.abs().sum(dim=0).bool()
        ok_states = states.abs().sum(dim=0).sum(dim=0).bool()

    # if we don't have any trajectories generated within the last 3 CFD episodes, it doesn't make sense to
    # continue with the training
    if cl[:, ok].size()[1] == 0:
        print("[env_model_rotating_cylinder.py]: could not find any valid trajectories from the last 3 CFD episodes!"
              "\nAborting training.")
        exit(0)

    # normalize the data to interval of [0, 1] (except alpha and beta)
    data["states"], data["min_max_states"] = normalize_data(states[:, :, ok_states])
    data["actions"], data["min_max_actions"] = normalize_data(actions[:, ok])
    data["cl"], data["min_max_cl"] = normalize_data(cl[:, ok])
    data["cd"], data["min_max_cd"] = normalize_data(cd[:, ok])
    data["alpha"], data["beta"], data["rewards"] = alpha[:, ok], beta[:, ok], rewards[:, ok]

    return data


def check_trajectories(cl: pt.Tensor, cd: pt.Tensor, actions: pt.Tensor, alpha: pt.Tensor, beta: pt.Tensor) -> Tuple:
    """
    check the model-generated trajectories wrt realistic values or nan's.

    Note: these boundaries depend on the current setup, e.g. Reynolds number and therefore may have to be modified

    :param cl: trajectory of cl
    :param cd: trajectory of cd
    :param actions: trajectory of actions
    :param alpha: trajectory of alpha
    :param beta: trajectory of beta
    :return: status if trajectory is valid or not and which parameter caused the issue as tuple: (status, param)
    """
    status = (True, None)
    if (pt.max(cl.abs()).item() > 1.3) or (pt.isnan(cl).any().item()):
        status = (False, "cl")
    elif (pt.max(cd).item() > 3.5) or (pt.min(cd).item() < 2.85) or (pt.isnan(cd).any().item()):
        status = (False, "cd")
    elif (pt.max(actions.abs()).item() > 5.0) or (pt.isnan(actions).any().item()):
        status = (False, "actions")
    elif (pt.max(alpha.abs()).item() > 5e3) or (pt.isnan(alpha).any().item()):
        status = (False, "alpha")
    elif (pt.max(beta.abs()).item() > 5e3) or (pt.isnan(beta).any().item()):
        status = (False, "beta")

    return status


def predict_trajectories(env_model_cl_p: list, env_model_cd: list, episode: int,
                         path: str, states: pt.Tensor, cd: pt.Tensor, cl: pt.Tensor, actions: pt.Tensor,
                         alpha: pt.Tensor, beta: pt.Tensor, n_probes: int, n_input_steps: int, min_max: dict,
                         len_trajectory: int = 400, model_no: int = None) -> dict and Tuple:
    """
    predict a trajectory based on a given initial state and action using trained environment models for cd, and cl-p

    :param env_model_cl_p: list containing the trained environment model ensemble for cl and p
    :param env_model_cd: list containing the trained environment model ensemble for cd
    :param episode: the current episode of the training
    :param path: path to the directory where the training is currently running
    :param states: pressure at probe locations sampled from trajectories generated by within CFD used as initial states
    :param cd: cd sampled from trajectories generated by within CFD used as initial states
    :param cl: cl sampled from trajectories generated by within CFD used as initial states
    :param actions: actions sampled from trajectories generated by within CFD used as initial states
    :param alpha: alpha values sampled from trajectories generated by within CFD used as initial states
    :param beta: beta values sampled from trajectories generated by within CFD used as initial states
    :param n_probes: number of probes places in the flow field
    :param n_input_steps: number as input time steps for the environment models
    :param min_max: the min- / max-values used for scaling the trajectories to the intervall [0, 1]
    :param len_trajectory: length of the trajectory, 1sec CFD = 100 epochs
    :param model_no: index of the model used for predictions, if None then each step a model is randomly chosen
    :return: the predicted trajectory and a tuple containing the status if the generated trajectory is within realistic
             bounds, and if status = False which parameter is out of bounds
    """

    # test model: loop over all test data and predict the trajectories based on given initial state and actions
    # for each model of the ensemble: load the current state dict
    for model in range(len(env_model_cl_p)):
        env_model_cl_p[model].load_state_dict(pt.load(join(path, "cl_p_model", f"bestModel_no{model}_val.pt")))
        env_model_cd[model].load_state_dict(pt.load(join(path, "cd_model", f"bestModel_no{model}_val.pt")))

    # load current policy network (saved at the end of the previous episode)
    policy_model = (pt.jit.load(open(join(path, f"policy_trace_{episode - 1}.pt"), "rb"))).eval()

    # use batch for prediction, because batch normalization only works for batch size > 1
    # -> at least 2 trajectories required
    batch_size, dev = 2, ["cuda" if pt.cuda.is_available() else "cpu"][0]
    shape = (batch_size, len_trajectory)
    traj_cd, traj_cl, traj_alpha, traj_beta, traj_actions, traj_p = pt.zeros(shape).to(dev), pt.zeros(shape).to(dev),\
                                                                    pt.zeros(shape).to(dev), pt.zeros(shape).to(dev),\
                                                                    pt.zeros(shape).to(dev), \
                                                                    pt.zeros((batch_size, len_trajectory, n_probes)).to(dev)
    for i in range(batch_size):
        traj_cd[i, :n_input_steps] = cd
        traj_cl[i, :n_input_steps] = cl
        traj_alpha[i, :n_input_steps] = alpha
        traj_beta[i, :n_input_steps] = beta
        traj_actions[i, :n_input_steps] = actions
        traj_p[i, :n_input_steps, :] = states

    # loop over the trajectory, each iteration move input window by one time step
    for t in range(len_trajectory - n_input_steps):
        # create the feature (same for both environment models)
        feature = pt.flatten(pt.concat([traj_p[:, t:t + n_input_steps, :],
                                        (traj_cl[:, t:t + n_input_steps]).reshape([batch_size, n_input_steps, 1]),
                                        traj_cd[:, t:t + n_input_steps].reshape([batch_size, n_input_steps, 1]),
                                        (traj_actions[:, t:t + n_input_steps]).reshape([batch_size, n_input_steps, 1])],
                                       dim=2),
                             start_dim=1).to(dev)

        if model_no is None:
            # randomly choose an environment model to make a prediction if no model is specified
            tmp_cd_model = env_model_cd[pt.randint(low=0, high=len(env_model_cl_p), size=(1, 1)).item()]
            tmp_cl_p_model = env_model_cl_p[pt.randint(low=0, high=len(env_model_cl_p), size=(1, 1)).item()]
        else:
            tmp_cd_model = env_model_cd[model_no]
            tmp_cl_p_model = env_model_cl_p[model_no]

        # make prediction for cd
        traj_cd[:, t + n_input_steps] = tmp_cd_model(feature).squeeze().detach()

        # make prediction for probes and cl
        prediction_cl_p = tmp_cl_p_model(feature).squeeze().detach()
        traj_p[:, t + n_input_steps, :] = prediction_cl_p[:, :n_probes]
        traj_cl[:, t + n_input_steps] = prediction_cl_p[:, -1]

        # use predicted (new) state to get an action for both environment models as new input
        # note: policy network uses real states as input (not scaled to [0, 1]), policy training currently on cpu
        s_real = denormalize_data(traj_p[:, t + n_input_steps, :], min_max["states"])
        tmp_pred = policy_model(s_real.to("cpu")).squeeze().detach()
        traj_alpha[:, t + n_input_steps], traj_beta[:, t + n_input_steps] = tmp_pred[:, 0], tmp_pred[:, 1]

        # sample the value for omega (scaled to [0, 1])
        beta_distr = pt.distributions.beta.Beta(traj_alpha[:, t + n_input_steps], traj_beta[:, t + n_input_steps])
        traj_actions[:, t + n_input_steps] = beta_distr.sample()

    # re-scale everything for PPO-training and sort into dict, therefore always use the first trajectory in the batch
    act_rescaled = denormalize_data(traj_actions, min_max["actions"])[0, :].to("cpu")
    cl_rescaled = denormalize_data(traj_cl, min_max["cl"])[0, :].to("cpu")
    cd_rescaled = denormalize_data(traj_cd, min_max["cd"])[0, :].to("cpu")
    p_rescaled = denormalize_data(traj_p, min_max["states"])[0, :, :].to("cpu")

    # sanity check if the created trajectories make sense
    status = check_trajectories(cl=cl_rescaled, cd=cd_rescaled, actions=act_rescaled, alpha=traj_alpha[0, :],
                                beta=traj_beta[0, :])

    # TODO: reward fct for fluidic pinball?
    output = {"states": p_rescaled, "cl": cl_rescaled, "cd": cd_rescaled, "alpha": traj_alpha[0, :].to("cpu"),
              "beta": traj_beta[0, :].to("cpu"), "actions": act_rescaled, "generated_by": "env_models",
              "rewards": 3.0 - (cd_rescaled + 0.1 * cl_rescaled.abs())}

    return output, status


def train_env_models(path: str, n_t_input: int, n_probes: int, data_cl_p: list, data_cd: list, n_neurons: int = 100,
                     n_layers: int = 3, n_neurons_cd: int = 50, n_layers_cd: int = 5, epochs: int = 2500,
                     epochs_cd: int = 2500, load: bool = False,
                     model_no: int = 0) -> EnvironmentModel and EnvironmentModel and list:
    """
    initializes two environment models, trains and validates them based on the sampled data from the CFD
    environment. The models are trained and validated using the previous 2 episodes run in the CFD environment

    :param path: path to the directory where the training is currently running
    :param n_t_input: number as input time steps for the environment models
    :param n_probes: number of probes places in the flow field
    :param data_cl_p: [dataloader_train, dataloader_val]
    :param data_cd: [dataloader_train, dataloader_val]
    :param n_neurons: number of neurons per layer for the cl-p-environment model
    :param n_layers: number of hidden layers for the cl-p-environment model
    :param n_neurons_cd: number of neurons per layer for the cd-environment model
    :param n_layers_cd: number of hidden layers for the cd-environment model
    :param epochs: number of epochs for training the cl-p-environment model
    :param epochs_cd: number of epochs for training the cd-environment model
    :param load: flag if models of last episodes should be used as initialization
    :param model_no: number of the environment model within the ensemble
    :return: both environment models (cl-p & cd), as well as the corresponding training- and validation losses
    """
    if not exists(path):
        mkdir(path)

    # train and validate environment models with CFD data from the previous episode
    print(f"start training the environment model no. {model_no} for cl & p")

    # initialize environment networks TODO: generalize n_actions, n_cl, n_cd -> fluidic pinball
    env_model_cl_p = EnvironmentModel(n_inputs=n_t_input * (n_probes + 3), n_outputs=n_probes + 1, n_neurons=n_neurons,
                                      n_layers=n_layers)
    env_model_cd = EnvironmentModel(n_inputs=n_t_input * (n_probes + 3), n_outputs=1, n_neurons=n_neurons_cd,
                                    n_layers=n_layers_cd)

    # move model to GPU if available
    device = ["cuda" if pt.cuda.is_available() else "cpu"][0]

    # load environment models trained in the previous CFD episode
    if load:
        env_model_cl_p.load_state_dict(pt.load(join(path, "cl_p_model", f"bestModel_no0_val.pt")))
        env_model_cd.load_state_dict(pt.load(join(path, "cd_model", f"bestModel_no0_val.pt")))

    # train environment models
    train_loss, val_loss = train_model(env_model_cl_p.to(device), dataloader_train=data_cl_p[0],
                                       dataloader_val=data_cl_p[1], save_dir=join(path, "cl_p_model"),
                                       epochs=epochs, save_name=f"bestModel_no{model_no}")

    print(f"starting training for cd model no. {model_no}")
    train_loss_cd, val_loss_cd = train_model(env_model_cd.to(device), dataloader_train=data_cd[0],
                                             dataloader_val=data_cd[1], save_name=f"bestModel_no{model_no}",
                                             epochs=epochs_cd, save_dir=join(path, "cd_model"))

    return env_model_cl_p, env_model_cd, [[train_loss, train_loss_cd], [val_loss, val_loss_cd]]


def print_trajectory_info(no: int, buffer_size: int, i: int, tra: dict, key: str) -> None:
    """
    if an invalid trajectory was generated this functions prints info's about the parameter which is out of bounds as
    well as min-/ max- and mean value of this parameter within the trajectory

    :param no: trajectory number wrt buffer size
    :param buffer_size: buffer size
    :param i: number of trails performed to generate the trajectory
    :param tra: the trajectory
    :param key: parameter which caused the out-of-bounds issue
    :return: None
    """
    vals = [(k, pt.min(tra[k]).item(), pt.max(tra[k]).item(), pt.mean(tra[k]).item()) for k in tra if k == key]
    print(f"\ndiscarding trajectory {no + 1}/{buffer_size} due to invalid values [try no. {i}]:")
    for val in vals:
        print(f"\tmin / max / mean {val[0]}: {round(val[1], 5)}, {round(val[2], 5)}, {round(val[3], 5)}")


def assess_model_performance(s_model: list, a_model: list, r_model: list, agent: PPOAgent) -> list:
    """
    computes the policy loss of the current MB-episode for each model in the ensemble

    :param s_model: predicted states by each environment model
    :param a_model: actions predicted by policy network for each environment model
    :param r_model: predicted rewards by each environment model
    :param agent: PPO-agent
    :return: policy loss wrt environment models
    """
    policy_loss = []

    # assess the policy loss for each model
    for m in range(a_model[0].size()[-1]):
        values = [agent._value(s[:, :, m]) for s in s_model]

        log_p_old = pt.cat([agent._policy.predict(s[:-1, :, m], a[:-1, m])[0] for s, a in zip(s_model, a_model)])
        gaes = pt.cat([compute_gae(r[:, m], v, agent._gamma, agent._lam) for r, v in zip(r_model, values)])
        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS_SP)

        states_wf = pt.cat([s[:-1, :, m] for s in s_model])
        actions_wf = pt.cat([a[:-1, m] for a in a_model])
        log_p_new, entropy = agent._policy.predict(states_wf, actions_wf)
        p_ratio = (log_p_new - log_p_old).exp()
        policy_objective = gaes * p_ratio
        policy_objective_clipped = gaes * p_ratio.clamp(1.0 - agent._policy_clip, 1.0 + agent._policy_clip)
        policy_loss.append(-pt.min(policy_objective, policy_objective_clipped).mean().item())

    return policy_loss


def fill_buffer_from_models(env_model_cl_p: list, env_model_cd: list, episode: int, path: str, observation: dict,
                            n_input: int, n_probes: int, buffer_size: int, len_traj: int,
                            agent: PPOAgent) -> Tuple[list, list]:
    """
    creates trajectories using data from the CFD environment as initial states and the previously trained environment
    models in order to fill the buffer

    :param env_model_cl_p: list with all trained environment models for cl and p
    :param env_model_cd: list with all trained environment models for cd
    :param episode: the current episode of the training
    :param path: path to the directory where the training is currently running
    :param observation: the trajectories sampled in CFD, used to sample input states for generating the trajectories
    :param n_input: number as input time steps for the environment models
    :param n_probes: number of probes places in the flow field
    :param buffer_size: size of the buffer, specified in args when running the run_training.py
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param agent: PPO-agent
    :return: a list with the length of the buffer size containing the generated trajectories
    """
    predictions, shape = [], (len_traj, len(env_model_cl_p))
    r_model_tmp, a_model_tmp, s_model_tmp = pt.zeros(shape), pt.zeros(shape), pt.zeros((shape[0], n_probes, shape[1]))
    r_model, a_model, s_model = [], [], []

    # min- / max-values used for normalization
    min_max = {"states": observation["min_max_states"], "cl": observation["min_max_cl"],
               "cd": observation["min_max_cd"], "actions": observation["min_max_actions"]}

    counter, failed, max_iter = 0, 0, 50
    while counter < buffer_size:
        print(f"start filling buffer with trajectory {counter + 1}/{buffer_size} using environment models")

        # for each trajectory sample input states from all available data within the CFD buffer
        no = pt.randint(low=0, high=observation["cd"].size()[1], size=(1, 1)).item()

        # then predict the trajectory (the env. models are loaded in predict trajectory function)
        pred, ok = predict_trajectories(env_model_cl_p, env_model_cd, episode, path, observation["states"][:, :, no],
                                        observation["cd"][:, no], observation["cl"][:, no],
                                        observation["actions"][:, no], observation["alpha"][:, no],
                                        observation["beta"][:, no], n_probes, n_input, min_max, len_traj)

        # only add trajectory to buffer if the values make sense, otherwise discard it
        if ok[0]:
            predictions.append(pred)

            # compute the uncertainty of the predictions for the rewards wrt the model number
            for model in range(len(env_model_cl_p)):
                tmp, _ = predict_trajectories(env_model_cl_p, env_model_cd, episode, path,
                                              observation["states"][:, :, no], observation["cd"][:, no],
                                              observation["cl"][:, no], observation["actions"][:, no],
                                              observation["alpha"][:, no], observation["beta"][:, no],
                                              n_probes, n_input, min_max, len_traj, model_no=model)
                r_model_tmp[:, model] = tmp["rewards"]
                a_model_tmp[:, model] = tmp["actions"]
                s_model_tmp[:, :, model] = tmp["states"]

            # same data structure as required in update-method of PPO-agent
            r_model.append(r_model_tmp)
            a_model.append(a_model_tmp)
            s_model.append(s_model_tmp)

            counter += 1
            failed = 0

        else:
            print_trajectory_info(counter, buffer_size, failed, pred, ok[1])
            failed += 1

        # if all the trajectories are invalid, abort training in order to avoid getting stuck in while-loop forever
        if failed >= max_iter:
            print(f"could not generate valid trajectories after {max_iter} iterations... going back to CFD")
            counter = buffer_size

    # compute the policy performance for each model in the current episode
    policy_loss_model = assess_model_performance(s_model, a_model, r_model, agent)

    return predictions, policy_loss_model


def generate_feature_labels(cd, states: pt.Tensor = None, actions: pt.Tensor = None, cl: pt.Tensor = None,
                            len_traj: int = 400, n_t_input: int = 30, n_probes: int = 12,
                            cd_model: bool = False) -> pt.Tensor or Tuple[pt.Tensor]:
    """
    create feature-label pairs of all available trajectories

    :param cd: trajectories of cd
    :param states: trajectories of probes, not required if feature-label pairs are created for cd-model
    :param actions: trajectories of actions, not required if feature-label pairs are created for cd-model
    :param cl: trajectories of cl, not required if feature-label pairs are created for cd-model
    :param len_traj: length of the trajectories
    :param n_t_input: number of input time steps for the environment model
    :param n_probes: number of probes placed in the flow field
    :param cd_model: flag weather feature-label pairs should be created for cd- or cl-p model
    :return: tensor with features and tensor with corresponding labels, sorted as [batches, N_features (or labels)]
    """

    # check if input corresponds to correct model
    input_types = [isinstance(states, type(None)), isinstance(actions, type(None)), isinstance(cl, type(None))]
    if True in input_types and cd_model is False:
        print("[env_rotating_cylinder.py]: can't generate features for cl-p environment models. States, actions and cl"
              "are not given!")
        exit(0)

    # TODO: generalize n_actions, n_cl, n_cd -> fluidic pinball
    n_traj, shape_input = cd.size()[1], (len_traj - n_t_input, n_t_input * (n_probes + 3))
    feature, label = [], []
    for n in range(n_traj):
        # tmp tensor for each trajectory: [N_features, N_input] -> model input always [batch_size, N_features]
        if cd_model:
            f, l = 0, pt.zeros(len_traj - n_t_input, 1)
        else:
            f, l = pt.zeros(shape_input), pt.zeros(len_traj - n_t_input, n_probes + 1)

        for t_idx in range(0, len_traj - n_t_input):
            # cd-models have the same feature as the cl-p models, therefore the feature don't need to be computed again
            if cd_model:
                l[t_idx, :] = cd[t_idx + n_t_input, n]
                f = 0

            else:
                # [n_probes * n_time_steps * states, n_time_steps * cl, n_time_steps * cd, n_time_steps * action]
                s = states[t_idx:t_idx + n_t_input, :, n].squeeze()
                cl_tmp = cl[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
                cd_tmp = cd[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
                a = actions[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
                f[t_idx, :] = pt.cat([s, cl_tmp, cd_tmp, a], dim=1).flatten()
                l[t_idx, :] = pt.cat([states[t_idx + n_t_input, :, n].squeeze(),
                                      cl[t_idx + n_t_input, n].unsqueeze(-1)], dim=0)
        feature.append(f)
        label.append(l)

    if cd_model:
        return pt.cat(label, dim=0)
    else:
        return pt.cat(feature, dim=0), pt.cat(label, dim=0)


def create_subset_of_data(data: TensorDataset, n_models: int, batch_size: int = 25) -> List[DataLoader]:
    """
    creates a subset of the dataset wrt number of models, so that each model can be trained on a different subset of the
    dataset in order to accelerate the overall training process

    :param data: the dataset consisting of features and labels
    :param n_models: number of models in the ensemble
    :param batch_size: batch size
    :return: list of the dataloaders created for each model
    """
    rest = len(data.indices) % (n_models - 1)
    idx = [int(len(data.indices) / (n_models - 1)) for _ in range(n_models-1)]

    # distribute the remaining idx equally over the models
    for i in range(rest):
        idx[i] += 1

    return [DataLoader(i, batch_size=batch_size, shuffle=True, drop_last=False) for i in random_split(data, idx)]


def wrapper_train_env_model_ensemble(train_path: str, cfd_obs: list, len_traj: int, n_states: int, buffer: int,
                                     n_models: int, n_time_steps: int = 30, e_re_train: int = 1000,
                                     e_re_train_cd: int = 1000, load: bool = False, n_layers_cl_p: int = 3,
                                     n_layers_cd: int = 5, n_neurons_cl_p: int = 100,
                                     n_neurons_cd: int = 50) -> Tuple[list, list, pt.Tensor, dict] \
                                                                or Tuple[list, list, list, dict]:
    """
    wrapper function for train the ensemble of environment models

    :param train_path: path to the directory where the training is currently running
    :param cfd_obs: list containing the file names with all episodes run in CFD so far
    :param len_traj: length of the trajectories
    :param n_states: number of probes places in the flow field
    :param buffer: buffer size
    :param n_models: number of environment models in the ensemble
    :param n_time_steps: number as input time steps for the environment models
    :param e_re_train:number of episodes for re-training the cl-p-models if no valid trajectories could be generated
    :param e_re_train_cd: number of episodes for re-training the cd-models if no valid trajectories could be generated
    :param load: flag if 1st model in ensemble is trained from scratch or if previous model is used as initialization
    :param n_layers_cl_p: number of neurons per layer for the cl-p-environment model
    :param n_neurons_cl_p: number of hidden layers for the cl-p-environment model
    :param n_neurons_cd: number of neurons per layer for the cd-environment model
    :param n_layers_cd: number of hidden layers for the cd-environment model
    :return: list with:
            [trained cl-p-ensemble, trained cd-ensemble, train- and validation losses, loaded trajectories from CFD]
    """
    cl_p_ensemble, cd_ensemble, losses = [], [], []
    obs = check_cfd_data(cfd_obs, len_traj=len_traj, n_probes=n_states, buffer_size=buffer, n_e_cfd=len(cfd_obs))

    # create feature-label pairs for all possible input states of given data
    features_cl_p, labels_cl_p = generate_feature_labels(states=obs["states"], cd=obs["cd"], actions=obs["actions"],
                                                         cl=obs["cl"], len_traj=len_traj, n_t_input=n_time_steps,
                                                         n_probes=n_states)

    labels_cd = generate_feature_labels(cd=obs["cd"], len_traj=len_traj, n_t_input=n_time_steps, n_probes=n_states,
                                        cd_model=True)

    # save first N time steps of CFD trajectories in order to sample initial states for MB-episodes
    init_data = {"min_max_states": obs["min_max_states"], "min_max_actions": obs["min_max_actions"],
                 "min_max_cl": obs["min_max_cl"], "min_max_cd": obs["min_max_cd"],
                 "alpha": obs["alpha"][:n_time_steps, :], "beta": obs["beta"][:n_time_steps, :],
                 "cl": obs["cl"][:n_time_steps, :], "cd": obs["cd"][:n_time_steps, :],
                 "actions": obs["actions"][:n_time_steps, :], "states": obs["states"][:n_time_steps, :, :],
                 "rewards": obs["rewards"][:n_time_steps, :]}

    # create dataset for both models -> features of cd-models the same as for cl-p-models
    device = ["cuda" if pt.cuda.is_available() else "cpu"][0]
    cl_p_data = TensorDataset(features_cl_p.to(device), labels_cl_p.to(device))
    cd_data = TensorDataset(features_cl_p.to(device), labels_cd.to(device))

    # split into training ind validation data -> 75% training, 25% validation data
    n_train = int(0.75 * features_cl_p.size()[0])
    n_val = features_cl_p.size()[0] - n_train
    train_cl_p, val_cl_p = random_split(cl_p_data, [n_train, n_val])
    train_cd, val_cd = random_split(cd_data, [n_train, n_val])

    # create dataloader and free up some memory
    loader_train = DataLoader(train_cl_p, batch_size=25, shuffle=True, drop_last=False)
    loader_val = DataLoader(val_cl_p, batch_size=25, shuffle=True, drop_last=False)
    loader_train_cd = DataLoader(train_cd, batch_size=25, shuffle=True, drop_last=False)
    loader_val_cd = DataLoader(val_cd, batch_size=25, shuffle=True, drop_last=False)

    del obs, features_cl_p, labels_cl_p, labels_cd

    # train 1st model in e = 0 with max. 2500 epochs, for e > 0: re-train from previous model with 1000 epochs
    if not load:
        env_model_cl_p, env_model_cd, loss = train_env_models(train_path, n_time_steps, n_states,
                                                              data_cl_p=[loader_train, loader_val],
                                                              data_cd=[loader_train_cd, loader_val_cd],
                                                              load=load, model_no=0, n_neurons=n_neurons_cl_p,
                                                              n_layers=n_layers_cl_p, n_neurons_cd=n_neurons_cd,
                                                              n_layers_cd=n_layers_cd)
    else:
        env_model_cl_p, env_model_cd, loss = train_env_models(train_path, n_time_steps, n_states,
                                                              data_cl_p=[loader_train, loader_val],
                                                              data_cd=[loader_train_cd, loader_val_cd],
                                                              load=True, model_no=0, epochs=e_re_train,
                                                              epochs_cd=e_re_train, n_neurons=n_neurons_cl_p,
                                                              n_layers=n_layers_cl_p, n_neurons_cd=n_neurons_cd,
                                                              n_layers_cd=n_layers_cd)

    # start filling the model ensemble "buffer"
    cl_p_ensemble.append(env_model_cl_p.eval())
    cd_ensemble.append(env_model_cd.eval())
    losses.append(loss)

    if n_models > 1:
        # train only 1st model on all the data, then initialize each new model with 1st model and train each model on
        # different subset of the data
        loader_train = create_subset_of_data(train_cl_p, n_models)
        loader_val = create_subset_of_data(val_cl_p, n_models)
        loader_train_cd = create_subset_of_data(train_cd, n_models)
        loader_val_cd = create_subset_of_data(val_cd, n_models)

        del train_cl_p, val_cl_p, train_cd, val_cd

        # TODO: parallelization when env = slurm -> availability GPU nodes...?
        for m in range(1, n_models):
            # train each new model in the ensemble initialized with the 1st model with max. 1000 epochs
            env_model_cl_p, env_model_cd, loss = train_env_models(train_path, n_time_steps, n_states,
                                                                  data_cl_p=[loader_train[m-1], loader_val[m-1]],
                                                                  data_cd=[loader_train_cd[m-1], loader_val_cd[m-1]],
                                                                  epochs=e_re_train, epochs_cd=e_re_train_cd, load=True,
                                                                  model_no=m, n_neurons=n_neurons_cl_p,
                                                                  n_layers=n_layers_cl_p, n_neurons_cd=n_neurons_cd,
                                                                  n_layers_cd=n_layers_cd)
            cl_p_ensemble.append(env_model_cl_p.eval())
            cd_ensemble.append(env_model_cd.eval())
            losses.append(loss)

    # in case only one model is used, then return the loss of the first model
    if len(losses) < 2:
        return cl_p_ensemble, cd_ensemble, losses[0], init_data
    else:
        return cl_p_ensemble, cd_ensemble, losses, init_data


if __name__ == "__main__":
    pass
