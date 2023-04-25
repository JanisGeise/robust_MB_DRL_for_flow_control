"""
 this script is responsible for everything related to the generation of trajectories using environment models, namely:
    - generating the trajectory
    - checking the trajectory for invalid values
    - assessing the model performance in order to determine if a switching back to CFD is required (switching criteria
      is implemented in 'env_model_rotating_cylinder.py')
"""
import os
import sys
import argparse
import torch as pt

from typing import Tuple
from os.path import join

BASE_PATH = os.environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import PPOAgent
from drlfoam.agent.agent import compute_gae
from drlfoam.constants import EPS_SP
from drlfoam.environment.env_model_rotating_cylinder import create_slurm_config
from drlfoam.environment.execute_prediction import predict_trajectories
from drlfoam.execution.manager import TaskManager
from drlfoam.execution.slurm import submit_and_wait


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
                            buffer_size: int, len_traj: int, agent: PPOAgent, env: str = "local") -> Tuple[list, list]:
    """
    creates trajectories using data from the CFD environment as initial states and the previously trained environment
    models in order to fill the buffer

    TODO: clean up implementation of this fct

    :param env_model: list with all trained environment models
    :param episode: the current episode of the training
    :param path: path to the directory where the training is currently running
    :param observation: the trajectories sampled in CFD, used to sample input states for generating the trajectories
    :param n_input: number as input time steps for the environment models
    :param n_probes: number of probes places in the flow field
    :param buffer_size: size of the buffer, specified in args when running the run_training.py
    :param len_traj: length of the trajectory, 1sec CFD = 100 epochs
    :param agent: PPO-agent
    :param env: environment, either 'local' or 'slurm', is set in 'run_training.py'
    :return: a list with the length of the buffer size containing the generated trajectories
    """
    predictions, shape = [], (len_traj, len(env_model))
    r_model_tmp, a_model_tmp, s_model_tmp = pt.zeros(shape), pt.zeros(shape), pt.zeros((shape[0], n_probes, shape[1]))
    r_model, a_model, s_model = [], [], []

    # min- / max-values used for normalization
    min_max = {"states": observation["min_max_states"], "cl": observation["min_max_cl"],
               "cd": observation["min_max_cd"], "actions": observation["min_max_actions"]}

    if env == "slurm":
        pt.save(observation, join(path, "obs_pred_traj.pt"))
        pt.save({"train_path": path, "env_model": env_model, "n_probes": n_probes, "n_input": n_input,
                 "episode": episode, "min_max": min_max, "len_traj": len_traj}, join(path, "settings_prediction.pt"))

        # write shell script for executing the prediction -> cwd on HPC = 'drlfoam/examples/'
        current_cwd = os.getcwd()
        config = create_slurm_config(case="pred_traj", exec_cmd="python3 execute_prediction.py -i $1 -n $2 -p $3")
        config.write(join(current_cwd, path, "execute_prediction.sh"))
        manager = TaskManager(n_runners_max=10)

    counter, failed_total, max_iter, n_failed_per_iter = 0, 0, 50, 0
    while counter < buffer_size:
        if env == "slurm":
            # in 1st iteration, we need to generate N_buffer trajectories
            if counter == 0:
                n_pred = buffer_size

            # in case some trajectories are invalid, we need to generate n_failed_per_iter trajectories more
            else:
                n_pred = n_failed_per_iter
                n_failed_per_iter = 0

            # go to training directory and execute the shell script for model training
            os.chdir(join(current_cwd, path))

            pred_id = []
            for m in range(n_pred):
                # for each trajectory sample input states from all available data within the CFD buffer
                no = pt.randint(low=0, high=observation["cd"].size()[1], size=(1, 1)).item()

                # random numbers are not unique -> use counter for saving / loading the data by name
                pred_id.append(m)

                manager.add(submit_and_wait, [f"execute_prediction.sh", str(m), str(no), path])
            manager.run()

            # then go back to the 'drlfoam/examples' directory and load the trajectories once all processes are done
            os.chdir(current_cwd)

            result = [pt.load(join(BASE_PATH, "examples", path, f"prediction_no{i}.pt")) for i in pred_id]
            for res in result:
                print(f"filling buffer with trajectory {counter + 1}/{buffer_size} using environment models")
                if res["ok"][0]:
                    predictions.append(res["pred"])
                    r_model.append(res["r_model"])
                    a_model.append(res["a_model"])
                    s_model.append(res["s_model"])
                    counter += 1
                else:
                    print_trajectory_info(counter, buffer_size, failed_total, res["pred"], res["ok"][1])
                    failed_total += 1
                    n_failed_per_iter += 1

            # clean up (tmp files)
            [os.remove(join(BASE_PATH, "examples", path, f"prediction_no{i}.pt")) for i in pred_id]

        else:
            print(f"filling buffer with trajectory {counter + 1}/{buffer_size} using environment models")

            # for each trajectory sample input states from all available data within the CFD buffer
            no = pt.randint(low=0, high=observation["cd"].size()[1], size=(1, 1)).item()

            # predict the trajectory (the env. models are loaded in predict trajectory function)
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
                failed_total = 0

            else:
                print_trajectory_info(counter, buffer_size, failed_total, pred, ok[1])
                failed_total += 1

        # if all the trajectories are invalid, abort training in order to avoid getting stuck in while-loop forever
        if failed_total >= max_iter:
            print(f"could not generate valid trajectories after {max_iter} iterations... going back to CFD")
            counter = buffer_size

    if failed_total < max_iter:
        # compute the policy performance for each model in the current episode if all trajectories are valid
        policy_loss_model = assess_model_performance(s_model, a_model, r_model, agent)
    else:
        # else go back to CFD -> policy loss usually in the order of 1e-17, so 1 is sufficient to trigger switching
        policy_loss_model = [1 for _ in range(len(env_model))]

    if env == "slurm":
        # final clean up
        os.remove(join(BASE_PATH, "examples", path, "settings_prediction.pt"))
        os.remove(join(BASE_PATH, "examples", path, "obs_pred_traj.pt"))
        os.remove(join(BASE_PATH, "examples", path, "execute_prediction.sh"))

    return predictions, policy_loss_model


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


if __name__ == "__main__":
    pass
