"""
    This script is implements the model-based part for the PPO-training routine. It contains the environment model class
    as well as functions for loading and sorting the trajectories, generating feature-label pairs and is responsible for
    managing the execution of the model-training.
"""
import os
import sys
import logging
import torch as pt

from time import time
from glob import glob
from os.path import join
from typing import Tuple, List
from os import chdir, getcwd, remove, environ
from torch.utils.data import DataLoader, TensorDataset, random_split

from drlfoam.environment.train_env_models import train_env_models, EnvironmentModel
from drlfoam.execution import SlurmConfig
from drlfoam.execution.manager import TaskManager
from drlfoam.execution.slurm import submit_and_wait

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

logging.basicConfig(level=logging.INFO)


class SetupEnvironmentModel:
    def __init__(self, n_models_better_perf: int = 3, n_models: int = 5, n_input_time_steps: int = 30, path: str = ""):
        """
        setup class for the environment models

        :param n_models_better_perf: number of models which have to improve the policy in order to not switch to CFD;
                                     this number is used to determine the threshold for the switching criteria
        :param n_models: number of models in the ensemble
        :param n_input_time_steps: number of subsequent time steps used for model input
        :param path: path to the training directory
        """
        self.path = path
        self.n_models = n_models
        self.t_input = n_input_time_steps
        self.obs_cfd = []
        self.len_traj = 200
        self.last_cfd = 0
        self.policy_loss = []
        # ensure that there are no round-off errors, which lead to switching
        self.threshold = round(n_models_better_perf / n_models, 3)
        self.start_training = None
        self._start_time = None
        self._time_cfd = []
        self._time_model_train = []
        self._time_prediction = []
        self._time_ppo = []

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
        # save train- and validation losses of the environment models (if losses are available)
        if self.n_models == 1:
            try:
                losses = {"train_loss": loss[0], "val_loss": loss[1]}
            except IndexError:
                losses = {"train_loss": [], "val_loss": []}
            self.save(episode, losses, name="env_model_loss")
        else:
            losses = {"train_loss": [l[0] for l in loss if l[0]], "val_loss": [l[1] for l in loss if l[0]]}
            self.save(episode, losses, name="env_model_loss")

    def save(self, episode, data, name: str = "observations"):
        pt.save(data, join(self.path, f"{name}_{episode}.pt"))

    def reset(self, episode):
        self.policy_loss = []
        self.last_cfd = episode

    def start_timer(self):
        self._start_time = time()

    def time_cfd_episode(self):
        self._time_cfd.append(time() - self._start_time)

    def time_model_training(self):
        self._time_model_train.append(time() - self._start_time)

    def time_mb_episode(self):
        self._time_prediction.append(time() - self._start_time)

    def time_ppo_update(self):
        self._time_ppo.append(time() - self._start_time)

    def compute_statistics(self, param):
        return [round(pt.mean(pt.tensor(param)).item(), 2), round(pt.std(pt.tensor(param)).item(), 2),
                round(pt.min(pt.tensor(param)).item(), 2), round(pt.max(pt.tensor(param)).item(), 2),
                round(sum(param) / (time() - self.start_training) * 100, 2)]

    def print_info(self):
        cfd = self.compute_statistics(self._time_cfd)
        model = self.compute_statistics(self._time_model_train)
        predict = self.compute_statistics(self._time_prediction)
        ppo = self.compute_statistics(self._time_ppo)

        # don't use logging here, because this is printed after the training is completed
        # -> so post-processing scripts don't need to be adapted
        print(f"time per CFD episode:\n\tmean: {cfd[0]}s\n\tstd: {cfd[1]}s\n\tmin: {cfd[2]}s\n\tmax: {cfd[3]}s\n\t"
              f"= {cfd[4]} % of total training time")
        print(f"time per model training:\n\tmean: {model[0]}s\n\tstd: {model[1]}s\n\tmin: {model[2]}s\n\tmax:"
              f" {model[3]}s\n\t= {model[4]} % of total training time")
        print(f"time per MB-episode:\n\tmean: {predict[0]}s\n\tstd: {predict[1]}s\n\tmin: {predict[2]}s\n\tmax:"
              f" {predict[3]}s\n\t= {predict[4]} % of total training time")
        print(f"time per update of PPO-agent:\n\tmean: {ppo[0]}s\n\tstd: {ppo[1]}s\n\tmin: {ppo[2]}s\n\tmax:"
              f" {ppo[3]}s\n\t= {ppo[4]} % of total training time")
        print(f"other: {round(100 - cfd[4] - model[4] - predict[4] - ppo[4], 2)} % of total training time")


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


def load_trajectory_data(files: list, len_traj: int, n_probes: int, n_actions: int) -> dict:
    """
    load the trajectory data from the observations_*.pt files

    :param files: list containing the file names of the last two episodes run in CFD environment
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param n_probes: number of probes placed in the flow field
    :param n_actions: number of actions
    :return: cl, cd, actions, states, alpha, beta
    """
    # in new version of drlfoam the observations are in stored in '.pt' files
    observations = [pt.load(open(file, "rb")) for file in files]

    # sort the trajectories from all workers, for training the models, it doesn't matter from which episodes the data is
    shape, n_col, data = (len_traj, len(observations) * len(observations[0])), 0, {}

    for key in observations[0][0].keys():
        # same data structure for states, actions, cl, cd, alpha & beta
        if key == "states":
            data[key] = pt.zeros((shape[0], shape[1], n_probes))
        elif key == "actions" and n_actions > 1:
            data[key] = pt.zeros((shape[0], shape[1], n_actions))
        else:
            data[key] = pt.zeros(shape)

    for observation in range(len(observations)):
        for j in range(len(observations[observation])):
            # omit failed or partly converged trajectories
            if not bool(observations[observation][j]) or observations[observation][j]["rewards"].size()[0] < len_traj:
                pass
            else:
                for key in data:
                    # in this case, only if n_actions > 1, this statement is false
                    if len(observations[observation][j][key].size()) < 2 and key != "states":
                        # sometimes the trajectories are 1 entry too long, in that case ignore the last value
                        data[key][:, n_col] = observations[observation][j][key][:len_traj]
                    else:
                        data[key][:, n_col, :] = observations[observation][j][key][:len_traj][:]
            n_col += 1

    return data


def check_cfd_data(files: list, len_traj: int, n_probes: int, buffer_size: int = 10, n_e_cfd: int = 0,
                   n_actions: int = 1) -> dict:
    """
    load the trajectory data, split the trajectories into training, validation- and test data (for sampling initial
    states), normalize all the data to [0, 1]

    :param files: list containing the file names of the last two episodes run in CFD environment
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param n_probes: number of probes placed in the flow field
    :param buffer_size: current buffer size
    :param n_e_cfd: number of currently available episodes run in CFD
    :param n_actions: amount of actions
    :return: dict containing the loaded, sorted and normalized data as well as the data for training- and validation
    """
    data = load_trajectory_data(files[-2:], len_traj, n_probes, n_actions)

    # delete the allocated columns of the failed trajectories (since they would alter the mean values)
    ok = data["rewards"].abs().sum(dim=0).bool()
    ok_states = data["states"].abs().sum(dim=0).sum(dim=1).bool()

    # if buffer of current CFD episode is empty, only one CFD episode is used for training, which is likely to crash.
    # In this case, load trajectories from 3 CFD episodes ago
    if len([i.item() for i in ok if i]) <= buffer_size and n_e_cfd >= 3:
        # determine current episode and then take the 'observations_*.pt' from 3 CFD episodes ago
        data_tmp = load_trajectory_data([files[-3]], len_traj, n_probes, n_actions)

        # merge into existing data -> structure always [len_traj, N_traj, (ggf. N_actions / N_states)]
        for key in data:
            data[key] = pt.concat([data[key], data_tmp[key]], dim=1)

        # and finally update the masks to remove non-converged trajectories
        ok = data["rewards"].abs().sum(dim=0).bool()
        ok_states = data["states"].abs().sum(dim=0).sum(dim=1).bool()

    # if we don't have any trajectories generated within the last 3 CFD episodes, it doesn't make sense to
    # continue with the training
    if data["rewards"][:, ok].size()[1] == 0:
        logging.critical("[env_model_rotating_cylinder.py]: could not find any valid trajectories from the last 3 CFD"
                         "episodes!\nAborting training.")
        exit(0)

    # add one dimension to all parameters, so in case of pinball the parameters can be merged into 1 tensor
    for key in ["states", "rewards", "actions", "alpha", "beta", "cl", "cd"]:
        # if action tensor 2D -> only 1 action -> cylinder2D -> need 1 additional dim
        # else the actions are already in the required shape of [len_traj, N_traj, N_actions]
        if key == "actions" and len(data["actions"].size()) <= 2:
            data[key].unsqueeze_(-1)

        # in case of fluidic pinball, the keys are named cx_*, cy_*, ... -> group them into tensors with same type
        elif key != "states" and key not in data.keys():
            if key == "cd":
                data[key] = pt.concat([data[k].unsqueeze(-1) for k in data.keys() if k.startswith("cx_")], dim=-1)
            elif key == "cl":
                data[key] = pt.concat([data[k].unsqueeze(-1) for k in data.keys() if k.startswith("cy_")], dim=-1)
            elif key == "alpha":
                data[key] = pt.concat([data[k].unsqueeze(-1) for k in data.keys() if k.startswith("alpha_")], dim=-1)
            elif key == "beta":
                data[key] = pt.concat([data[k].unsqueeze(-1) for k in data.keys() if k.startswith("beta_")], dim=-1)

        # otherwise we have cylinder2D -> we need 1 additional dim for cl, cd, alpha & beta
        elif key != "states" and key != "rewards" and {"alpha", "beta", "cl", "cd"}.issubset(set(data.keys())):
            data[key].unsqueeze_(-1)
        else:
            continue

    # scale to interval of [0, 1] (except alpha and beta), use list() to avoid runtimeError when deleting the keys
    for key in list(data.keys()):
        # delete all keys that are no longer required
        if key.endswith("_a") or key.endswith("_b") or key.endswith("_c"):
            data.pop(key)
        elif key != "rewards" and key != "alpha" and key != "beta":
            # all parameters except the rewards have 1 additional dim -> so use the mask for the states
            data[key], data[f"min_max_{key}"] = normalize_data(data[key][:, ok_states, :])
        elif key != "alpha" and key != "beta":
            data[key], data[f"min_max_{key}"] = normalize_data(data[key][:, ok])
        else:
            # alpha and bet don't need to be re-scaled since they are always in [0, 1]
            data[key] = data[key][:, ok]

    return data


def generate_feature_labels(cd, states: pt.Tensor, actions: pt.Tensor, cl: pt.Tensor,
                            n_t_input: int = 30) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    create feature-label pairs of all available trajectories

    :param cd: trajectories of cd
    :param states: trajectories of probes, not required if feature-label pairs are created for cd-model
    :param actions: trajectories of actions, not required if feature-label pairs are created for cd-model
    :param cl: trajectories of cl, not required if feature-label pairs are created for cd-model
    :param n_t_input: number of input time steps for the environment model
    :return: tensor with features and tensor with corresponding labels, sorted as [batches, N_features (or labels)]
    """
    n_actions, n_cl, n_cd, n_probes = actions.size()[-1], cl.size()[-1], cd.size()[-1], states.size()[-1]
    len_traj, feature, label = cd.size()[0], [], []
    n_traj, shape_input = cd.size()[1], (len_traj - n_t_input, n_t_input * (n_probes + n_cl + n_cd + n_actions))
    for n in range(n_traj):
        f, l = pt.zeros(shape_input), pt.zeros(len_traj - n_t_input, n_probes + n_cl + n_cd)
        for t_idx in range(len_traj - n_t_input):
            # [n_probes * n_time_steps * states, n_time_steps * cl, n_time_steps * cd, n_time_steps * action]
            s = states[t_idx:t_idx + n_t_input, n, :].squeeze()
            cl_tmp = cl[t_idx:t_idx + n_t_input, n]
            cd_tmp = cd[t_idx:t_idx + n_t_input, n]
            a = actions[t_idx:t_idx + n_t_input, n]
            f[t_idx, :] = pt.cat([s, cl_tmp, cd_tmp, a], dim=1).flatten()
            l[t_idx, :] = pt.cat([states[t_idx + n_t_input, n, :].squeeze(), cl[t_idx + n_t_input, n],
                                  cd[t_idx + n_t_input, n]], dim=0)
        feature.append(f)
        label.append(l)

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
    rest = len(data.indices) % n_models
    idx = [int(len(data.indices) / n_models) for _ in range(n_models)]

    # distribute the remaining idx equally over the models
    for i in range(rest):
        idx[i] += 1

    return [DataLoader(i, batch_size=batch_size, shuffle=True, drop_last=False) for i in random_split(data, idx)]


def create_slurm_config(case: str, exec_cmd: str, aws: bool = False) -> SlurmConfig:
    """
    creates SLURM config for executing model training & prediction of MB-trajectories in parallel

    :param case: job name
    :param exec_cmd: python script which shall be executed plus optional parameters -> this str is executed by slurm
    :param aws: in case the training is done on AWS, the config looks slightly different (here just an example),
                the parameter aws=True needs to be set in:
                 - 'env_model_rotating_cylinder.py', fct. 'wrapper_train_env_model_ensemble' (line 428)
                 - 'predict_trajectories.py', fct. 'fill_buffer_from_models' (line 94)
    :return: slurm config
    """
    if pt.cuda.is_available():
        partition = "gpu02_queue"
        task_per_node = 1
    else:
        partition = "standard"
        task_per_node = 4
    if aws:
        slurm_config = SlurmConfig(partition="queue-1", n_nodes=1, n_tasks_per_node=8, job_name=case,
                                   modules=["openmpi/4.1.5"], time="00:10:00", constraint="c5a.24xlarge",
                                   commands=[f"source /{join('fsx', 'OpenFOAM', 'OpenFOAM-v2206', 'etc', 'bashrc')}",
                                             f"source /{join('fsx', 'drlfoam', 'setup-env')}",
                                             f"cd /{join('fsx', 'drlfoam', 'drlfoam', 'environment')}", exec_cmd])
    else:
        slurm_config = SlurmConfig(partition=partition, n_nodes=1, n_tasks_per_node=task_per_node, job_name=case,
                                   modules=["python/3.8.2"], time="00:15:00",
                                   commands=[f"source {join('~', 'drlfoam', 'pydrl', 'bin', 'activate')}",
                                             f"source {join('~', 'drlfoam', 'setup-env --container')}",
                                             f"cd {join('~', 'drlfoam', 'drlfoam', 'environment')}", exec_cmd])

    # add number of GPUs and write script to cwd
    if pt.cuda.is_available():
        slurm_config._options["--gres"] = "gpu:1"

    return slurm_config


def wrapper_train_env_model_ensemble(train_path: str, cfd_obs: list, len_traj: int, n_states: int, buffer: int,
                                     n_models: int, n_time_steps: int = 30, e_re_train: int = 1000,
                                     load: bool = False, env: str = "local", n_layers: int = 3,
                                     n_neurons: int = 100,
                                     n_actions: int = 1) -> Tuple[list, pt.Tensor, dict] or Tuple[list, list, dict]:
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
    :param load: flag if 1st model in ensemble is trained from scratch or if previous model is used as initialization
    :param env: environment, either 'local' or 'slurm', is set in 'run_training.py'
    :param n_layers: number of neurons per layer for the environment model
    :param n_neurons: number of hidden layers for the environment model
    :param n_actions: number of actions
    :return: trained model-ensemble, train- and validation losses, initial states for the following MB-episodes
    """
    model_ensemble, losses = [], []
    obs = check_cfd_data(cfd_obs, len_traj=len_traj, n_probes=n_states, buffer_size=buffer, n_e_cfd=len(cfd_obs),
                         n_actions=n_actions)

    # get n_cl and n_cd, because they may differ from n_actions in the future
    n_cl, n_cd = obs["cl"].size()[-1], obs["cd"].size()[-1]

    # create feature-label pairs for all possible input states of given data
    features, labels = generate_feature_labels(states=obs["states"], cd=obs["cd"], actions=obs["actions"],
                                               cl=obs["cl"], n_t_input=n_time_steps)

    # save first N time steps of CFD trajectories in order to sample initial states for MB-episodes
    init_data = {"min_max_states": obs["min_max_states"], "min_max_actions": obs["min_max_actions"],
                 "min_max_cl": obs["min_max_cl"], "min_max_cd": obs["min_max_cd"],
                 "alpha": obs["alpha"][:n_time_steps, :, :], "beta": obs["beta"][:n_time_steps, :, :],
                 "cl": obs["cl"][:n_time_steps, :, :], "cd": obs["cd"][:n_time_steps, :, :],
                 "actions": obs["actions"][:n_time_steps, :, :], "states": obs["states"][:n_time_steps, :, :],
                 "rewards": obs["rewards"][:n_time_steps, :]}

    # create dataset for both models -> features of cd-models the same as for cl-p-models
    device = "cuda" if pt.cuda.is_available() else "cpu"
    data = TensorDataset(features.to(device), labels.to(device))

    # split into training ind validation data -> 75% training, 25% validation data
    n_train = int(0.75 * features.size()[0])
    n_val = features.size()[0] - n_train
    train, val = random_split(data, [n_train, n_val])

    del obs, features, labels

    # initialize environment networks
    env_model = EnvironmentModel(n_inputs=n_time_steps * (n_states + n_cl + n_cd + n_actions),
                                 n_outputs=n_states + n_cl + n_cd, n_neurons=n_neurons, n_layers=n_layers)

    # train each model on different subset of the data. In case only 1 model is used, then this model is trained on
    # complete dataset
    loader_train = create_subset_of_data(train, n_models)
    loader_val = create_subset_of_data(val, n_models)

    del train, val

    if env == "local":
        for m in range(n_models):
            # initialize each model with different seed value
            pt.manual_seed(m)
            if pt.cuda.is_available():
                pt.cuda.manual_seed_all(m)

            # (re-) train each model in the ensemble with max. (1000) 2500 epochs
            if not load:
                loss = train_env_models(train_path, env_model, data=[loader_train[m], loader_val[m]], model_no=m)
            else:
                loss = train_env_models(train_path, env_model, data=[loader_train[m], loader_val[m]], load=True,
                                        model_no=m, epochs=e_re_train)
            model_ensemble.append(env_model.eval())
            losses.append(loss)
    else:
        pt.save(loader_train, join(train_path, "loader_train.pt"))
        pt.save(loader_val, join(train_path, "loader_val.pt"))
        pt.save({"train_path": train_path, "env_model": env_model, "epochs": e_re_train, "load": load},
                join(train_path, "settings_model_training.pt"))

        # write shell script for executing the model training -> on HPC = '~/drlfoam/examples/' instead of '/examples/'
        current_cwd = getcwd()

        # on AWS, cwd should start with /fsx/, e.g. '/fsx/drlfoam/'
        if current_cwd.startswith("/fsx/"):
            aws = True
        else:
            aws = False
        config = create_slurm_config(case="model_train", exec_cmd="python3 train_env_models.py -m $1 -p $2", aws=aws)
        config.write(join(current_cwd, train_path, "execute_model_training.sh"))
        manager = TaskManager(n_runners_max=10)

        # go to training directory and execute the shell script for model training
        chdir(join(current_cwd, train_path))
        for m in range(n_models):
            manager.add(submit_and_wait, [f"execute_model_training.sh", str(m), train_path])
        manager.run()

        # then go back to the 'drlfoam/examples' directory and continue training
        chdir(current_cwd)

        for m in range(n_models):
            # update path (relative path not working on cluster)
            train_path = join(BASE_PATH, "examples", train_path)

            # load the losses once the training is done -> in case job gets canceled there is no loss available
            model_ensemble.append(env_model.eval())
            try:
                losses.append([pt.load(join(train_path, "env_model", f"loss{m}_train.pt")),
                               pt.load(join(train_path, "env_model", f"loss{m}_val.pt"))])
            except FileNotFoundError:
                losses.append([[], []])

        # clean up
        [remove(f) for f in glob(join(train_path, "loader_*.pt"))]
        [remove(f) for f in glob(join(train_path, "env_model", "loss*.pt"))]
        remove(join(train_path, "settings_model_training.pt"))
        remove(join(train_path, "execute_model_training.sh"))

    # in case only one model is used, then return the loss of the first model
    if len(losses) < 2:
        return model_ensemble, losses[0], init_data
    else:
        return model_ensemble, losses, init_data


def check_finish_time(base_path: str, t_end: int, environment: str) -> None:
    """
    checks if the user-specified finish time is greater than the end time of the base case, if not then exit with error
    message

    :param base_path: BASE_PATH defined in run_training
    :param t_end: user-specified finish time
    :param environment: environment, either rotatingCylinder2D or rotatingPinball2D
    :return: None
    """
    pwd = join(base_path, "openfoam", "test_cases", environment, "system", "controlDict")
    with open(pwd, "r") as f:
        lines = f.readlines()

    # get the end time of the base case, normally endTime is specified in l. 28, but in case of modifications, check
    # lines 20-35
    t_base = [float(i.strip(";\n").split(" ")[-1]) for i in lines[20:35] if i.startswith("endTime")][0]

    if t_base >= t_end:
        logging.critical(f"specified finish time is smaller than end time of base case! The finish time needs to be "
                         f"greater than {t_base}. Exiting...")
        exit(0)


if __name__ == "__main__":
    pass
