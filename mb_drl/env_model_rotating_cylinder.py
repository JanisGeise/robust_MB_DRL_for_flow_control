"""
    This script is implements the model-based part for the PPO-training routine. It contains the environment model class
    as well as functions for loading and sorting the trajectories, generating feature-label pairs and is responsible for
    managing the execution of the model-training.
"""
import sys
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
                losses = {"train_loss": loss[0][0], "val_loss": loss[1][0]}
            except IndexError:
                losses = {"train_loss": [], "val_loss": []}
            self.save(episode, losses, name="env_model_loss")
        else:
            losses = {"train_loss": [l[0][0] for l in loss if l[0]], "val_loss": [l[0][1] for l in loss if l[0]]}
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


def generate_feature_labels(cd, states: pt.Tensor = None, actions: pt.Tensor = None, cl: pt.Tensor = None,
                            len_traj: int = 400, n_t_input: int = 30, n_probes: int = 12) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    create feature-label pairs of all available trajectories

    :param cd: trajectories of cd
    :param states: trajectories of probes, not required if feature-label pairs are created for cd-model
    :param actions: trajectories of actions, not required if feature-label pairs are created for cd-model
    :param cl: trajectories of cl, not required if feature-label pairs are created for cd-model
    :param len_traj: length of the trajectories
    :param n_t_input: number of input time steps for the environment model
    :param n_probes: number of probes placed in the flow field
    :return: tensor with features and tensor with corresponding labels, sorted as [batches, N_features (or labels)]
    """

    # check if input corresponds to correct model
    input_types = [isinstance(states, type(None)), isinstance(actions, type(None)), isinstance(cl, type(None))]
    if True in input_types:
        print("[env_rotating_cylinder.py]: can't generate features for cl-p environment models. States, actions and cl"
              "are not given!")
        exit(0)

    n_actions, n_cl, n_cd = 1, 1, 1     # TODO: this will be replaced by actual n_* later -> fluidic pinball
    n_traj, shape_input = cd.size()[1], (len_traj - n_t_input, n_t_input * (n_probes + n_cl + n_cd + n_actions))
    feature, label = [], []
    for n in range(n_traj):
        f, l = pt.zeros(shape_input), pt.zeros(len_traj - n_t_input, n_probes + n_cl + n_cd)
        for t_idx in range(0, len_traj - n_t_input):
            # [n_probes * n_time_steps * states, n_time_steps * cl, n_time_steps * cd, n_time_steps * action]
            s = states[t_idx:t_idx + n_t_input, :, n].squeeze()
            cl_tmp = cl[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
            cd_tmp = cd[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
            a = actions[t_idx:t_idx + n_t_input, n].unsqueeze(-1)
            f[t_idx, :] = pt.cat([s, cl_tmp, cd_tmp, a], dim=1).flatten()
            l[t_idx, :] = pt.cat([states[t_idx + n_t_input, :, n].squeeze(), cl[t_idx + n_t_input, n].unsqueeze(-1),
                                  cd[t_idx + n_t_input, n].unsqueeze(-1)], dim=0)
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
        slurm_config = SlurmConfig(partition="queue-1", n_nodes=1, n_tasks_per_node=10, job_name=case,
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
                                     n_neurons: int = 100) -> Tuple[list, pt.Tensor, dict] or Tuple[list, list, dict]:
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
    :return: trained model-ensemble, train- and validation losses, initial states for the following MB-episodes
    """
    model_ensemble, losses = [], []
    obs = check_cfd_data(cfd_obs, len_traj=len_traj, n_probes=n_states, buffer_size=buffer, n_e_cfd=len(cfd_obs))

    # create feature-label pairs for all possible input states of given data
    features, labels = generate_feature_labels(states=obs["states"], cd=obs["cd"], actions=obs["actions"],
                                               cl=obs["cl"], len_traj=len_traj, n_t_input=n_time_steps,
                                               n_probes=n_states)

    # save first N time steps of CFD trajectories in order to sample initial states for MB-episodes
    init_data = {"min_max_states": obs["min_max_states"], "min_max_actions": obs["min_max_actions"],
                 "min_max_cl": obs["min_max_cl"], "min_max_cd": obs["min_max_cd"],
                 "alpha": obs["alpha"][:n_time_steps, :], "beta": obs["beta"][:n_time_steps, :],
                 "cl": obs["cl"][:n_time_steps, :], "cd": obs["cd"][:n_time_steps, :],
                 "actions": obs["actions"][:n_time_steps, :], "states": obs["states"][:n_time_steps, :, :],
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
    n_cl, n_cd, n_actions = 1, 1, 1     # TODO: this will be replaced by actual n_* later -> fluidic pinball
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

        # write shell script for executing the model training -> cwd on HPC = 'drlfoam/examples/'
        current_cwd = getcwd()
        config = create_slurm_config(case="model_train", exec_cmd="python3 train_env_models.py -m $1 -p $2")
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


if __name__ == "__main__":
    pass
