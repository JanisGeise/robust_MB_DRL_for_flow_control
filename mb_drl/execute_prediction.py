
import os
import sys
import argparse
import torch as pt

from typing import Tuple
from os.path import join

BASE_PATH = os.environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.environment.env_model_rotating_cylinder import denormalize_data


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

        # make prediction for probes, cl, and cd TODO: generalize for different n_cl, n_cd, n_actions
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


def execute_prediction_slurm(pred_id: int, no: int, train_path: str = "examples/run_training/") -> None:
    """
    executes the prediction of the MB-trajectories on an HPC cluster using the singularity container

    :param pred_id:
    :param no: number of the trajectory containing the initial states
    :param train_path: path to current PPO-training directory
    :return: None
    """
    # cwd = 'drlfoam/drlfoam/environment/', so go back to the training directory
    os.chdir(join("..", "..", "examples"))

    # update path (relative path not working on cluster)
    train_path = join(BASE_PATH, "examples", train_path)

    settings = pt.load(join(train_path, "settings_prediction.pt"))
    trajectories = pt.load(join(train_path, "obs_pred_traj.pt"))

    shape = (settings["len_traj"], len(settings["env_model"]))
    r_model, a_model, s_model = pt.zeros(shape), pt.zeros(shape), pt.zeros((shape[0], settings["n_probes"], shape[1]))

    pred, ok = predict_trajectories(settings["env_model"], settings["episode"], train_path,
                                    trajectories["states"][:, :, no],
                                    trajectories["cd"][:, no], trajectories["cl"][:, no],
                                    trajectories["actions"][:, no], trajectories["alpha"][:, no],
                                    trajectories["beta"][:, no], settings["n_probes"], settings["n_input"],
                                    settings["min_max"], settings["len_traj"])

    # only add trajectory to buffer if the values make sense, otherwise discard it
    if ok[0]:
        # compute the uncertainty of the predictions for the rewards wrt the model number
        for model in range(len(settings["env_model"])):
            tmp, _ = predict_trajectories(settings["env_model"], settings["episode"], train_path,
                                          trajectories["states"][:, :, no],
                                          trajectories["cd"][:, no], trajectories["cl"][:, no],
                                          trajectories["actions"][:, no], trajectories["alpha"][:, no],
                                          trajectories["beta"][:, no], settings["n_probes"], settings["n_input"],
                                          settings["min_max"], settings["len_traj"], model_no=model)
            r_model[:, model] = tmp["rewards"]
            a_model[:, model] = tmp["actions"]
            s_model[:, :, model] = tmp["states"]

    # print some info to the slurm*.out, so the log can be assigned to this process
    print(f"DEBUG: episode no. {settings['episode']}, trajectory no. {pred_id}, valid trajectory: {ok[0]}")

    # save the trajectory, status and uncertainty wrt model
    pt.save({"pred": pred, "ok": ok, "r_model": r_model, "s_model": s_model, "a_model": a_model},
            join(train_path, f"prediction_no{pred_id}.pt"))


if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-i", "--id", required=True, help="process id")
    ag.add_argument("-n", "--number", required=True, help="number of the trajectory containing the initial states")
    ag.add_argument("-p", "--path", required=True, type=str, help="path to training directory")
    args = ag.parse_args()
    execute_prediction_slurm(int(args.id), int(args.number), args.path)
