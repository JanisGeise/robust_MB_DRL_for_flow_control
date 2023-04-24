"""
    TODO: update doc

    This script is implements the model-based part for the PPO-training routine. It contains the environment model class
    as well as functions for loading and sorting the trajectories, training the model ensembles and generating the
    model-based trajectories for the PPO-training.
"""
import sys
import torch as pt

from time import time
from glob import glob
from os.path import join
from typing import Tuple, List
from os import chdir, getcwd, remove, environ
from torch.utils.data import DataLoader, TensorDataset, random_split

from drlfoam.agent import PPOAgent
from drlfoam.agent.agent import compute_gae
from drlfoam.constants import EPS_SP

from drlfoam.environment.train_env_models import train_env_models, EnvironmentModel
from drlfoam.execution import SlurmConfig
from drlfoam.execution.manager import TaskManager
from drlfoam.execution.slurm import submit_and_wait

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)


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


def predict_trajectories(env_model: list, episode: int, path: str, states: pt.Tensor, cd: pt.Tensor, cl: pt.Tensor,
                         actions: pt.Tensor, alpha: pt.Tensor, beta: pt.Tensor, n_probes: int, n_input_steps: int,
                         min_max: dict, len_trajectory: int = 400, model_no: int = None) -> dict and Tuple:
    """
    predict a trajectory based on a given initial state and action using trained environment models for cd, and cl-p

    :param env_model: list containing the trained environment model ensemble
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
    for model in range(len(env_model)):
        env_model[model].load_state_dict(pt.load(join(path, "env_model", f"bestModel_no{model}_val.pt")))

    # load current policy network (saved at the end of the previous episode)
    policy_model = (pt.jit.load(open(join(path, f"policy_trace_{episode - 1}.pt"), "rb"))).eval()

    # use batch for prediction, because batch normalization only works for batch size > 1
    # -> at least 2 trajectories required
    batch_size, dev = 2, "cuda" if pt.cuda.is_available() else "cpu"
    shape = (batch_size, len_trajectory)
    traj_cd, traj_cl, traj_alpha, traj_beta, traj_actions, traj_p = pt.zeros(shape).to(dev), pt.zeros(shape).to(dev), \
                                                                    pt.zeros(shape).to(dev), pt.zeros(shape).to(dev), \
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
            tmp_env_model = env_model[pt.randint(low=0, high=len(env_model), size=(1, 1)).item()]
        else:
            tmp_env_model = env_model[model_no]

        # make prediction for probes, cl, and cd
        prediction = tmp_env_model(feature).squeeze().detach()
        traj_p[:, t + n_input_steps, :] = prediction[:, :n_probes]
        traj_cl[:, t + n_input_steps] = prediction[:, -2]
        traj_cd[:, t + n_input_steps] = prediction[:, -1]

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

    # TODO: add reward fct for fluidic pinball -> choose reward fct based on current environment
    output = {"states": p_rescaled, "cl": cl_rescaled, "cd": cd_rescaled, "alpha": traj_alpha[0, :].to("cpu"),
              "beta": traj_beta[0, :].to("cpu"), "actions": act_rescaled, "generated_by": "env_models",
              "rewards": 3.0 - (cd_rescaled + 0.1 * cl_rescaled.abs())}

    return output, status


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


def fill_buffer_from_models(env_model: list, episode: int, path: str, observation: dict, n_input: int, n_probes: int,
                            buffer_size: int, len_traj: int, agent: PPOAgent) -> Tuple[list, list]:
    """
    creates trajectories using data from the CFD environment as initial states and the previously trained environment
    models in order to fill the buffer

    :param env_model: list with all trained environment models
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
    predictions, shape = [], (len_traj, len(env_model))
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
        # TODO start: parallelize predictions when running training on cluster
        #               1. save obs
        #               2. each proc predicts traj -> param 'no' for sampling init states
        #               3. if traj valid then compute r_model_tmp, ...
        #               4. else save invalid traj & 'ok' param
        #               5. read in all the data once all procs finished -> if valid: append pred, *_model_tmp, else:
        #               6. count all invalid traj, print the info to log, set failed = N_discarded_traj
        #               7. start N_discard new procs and repeat until enough traj generated or failed >= max_iter
        #               8. either go back to CFD or asses model_performance
        pred, ok = predict_trajectories(env_model, episode, path, observation["states"][:, :, no],
                                        observation["cd"][:, no], observation["cl"][:, no],
                                        observation["actions"][:, no], observation["alpha"][:, no],
                                        observation["beta"][:, no], n_probes, n_input, min_max, len_traj)

        # only add trajectory to buffer if the values make sense, otherwise discard it
        if ok[0]:
            predictions.append(pred)

            # compute the uncertainty of the predictions for the rewards wrt the model number
            for model in range(len(env_model)):
                tmp, _ = predict_trajectories(env_model, episode, path,
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

            # TODO end

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


def create_slurm_config(case: str, exec_cmd: str) -> SlurmConfig:
    """
    creates SLURM config for executing model training & prediction of MB-trajectories in parallel

    :param case: job name
    :param exec_cmd: python script which shall be executed plus optional parameters -> this str is executed by slurm
    :return: slurm config
    """
    if pt.cuda.is_available():
        partition = "gpu02_queue"
        task_per_node = 1
    else:
        partition = "standard"
        task_per_node = 4
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
