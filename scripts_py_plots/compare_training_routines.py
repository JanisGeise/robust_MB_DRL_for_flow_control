"""
    brief:
        - this script compares the prediction accuracy of the two different MB-training routines currently implemented
          with the MF-trajectories, without the necessity to run a full MB-training since the prediction is done using
          available MF-training data

    dependencies:
        - 'plot_ppo_results.py' for plotting the training- and validation losses
        - 'env_model_rotating_cylinder.py' & 'env_model_rotating_cylinder_new_training_routine.py' as training routines
          which are to be compared (MB-DRL)

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import pickle
import matplotlib.pyplot as plt

import torch as pt
from shutil import rmtree
from os import path, mkdir
from torch import manual_seed
from typing import List, Tuple

from plot_ppo_results import plot_train_validation_loss
from mb_drl.env_model_rotating_cylinder import denormalize_data, normalize_data
from mb_drl.env_model_rotating_cylinder import wrapper_train_env_model_ensemble as original_training_routine
from mb_drl.env_model_rotating_cylinder_new_training_routine import wrapper_train_env_model_ensemble as new_training_routine


def plot_trajectories_of_probes(path: str, real_data: dict, predicted_data: List[dict], color: list, legend: list,
                                n_probes: int = 12, episode: int = 0) -> None:
    """
    plots the probe trajectory of the probes, sampled from the CFD environment in comparison to the predicted trajectory
    by the environment model

    :param path: setup containing all the paths etc.
    :param real_data: states (trajectory of probes) sampled in the CFD environment
    :param predicted_data: states (trajectory of probes) predicted by the environment model
    :param color: line colors for the plot
    :param legend: legend entries
    :param n_probes: total number of probes
    :param episode: number of episode (only used as save name for plot if parameter = "episodes")
    :return: None
    """
    x = range(predicted_data[0]["states"].size()[0])
    fig, ax = plt.subplots(nrows=n_probes, ncols=1, figsize=(9, 9), sharex="all")
    for i in range(n_probes):
        for j in range(len(predicted_data) + 1):
            if j == 0:
                ax[i].plot(x, real_data["states"][:, i], color="black", label=legend[j])
            else:
                ax[i].plot(x, predicted_data[j-1]["states"][:, i], color=color[j-1], label=legend[j])
            ax[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=40, usetex=True, fontsize=13)
        if i <= 2:
            ax[i].set_ylim(-1.0, 0.5)
        else:
            ax[i].set_ylim(-0.75, 0.75)

    fig.subplots_adjust(hspace=0.75)
    ax[-1].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, top=0.90)
    plt.savefig("".join([path, f"/plots/real_trajectories_vs_prediction_episode{episode}.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_vs_prediction(path: str, real_data: dict, predicted_data: List[dict], color: list, legend: list,
                             episode: int = 0) -> None:
    """
    plots the trajectory of cl and cd, sampled from the CFD environment in comparison to the predicted trajectory by the
    environment model

    :param path: save path
    :param real_data: trajectory of cl and cd sampled in the CFD environment
    :param predicted_data: trajectory of cl and cd predicted by the environment models
    :param color: line colors for the plot
    :param legend: legend entries
    :param episode: number of the episode (only used for the save name of plot)
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(len(predicted_data)):
        x = range(predicted_data[i]["cl"].size()[0])
        for j in range(len(predicted_data) + 1):
            if i == 0:
                if j == 0:
                    ax[i].plot(x, real_data["cl"], color="black", label=legend[j])
                else:
                    ax[i].plot(x, predicted_data[j-1]["cl"], color=color[j-1], label=legend[j])
                ax[i].set_ylabel("$lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)
                ax[i].set_ylim(-1.25, 1.25)
            else:
                if j == 0:
                    ax[i].plot(x, real_data["cd"], color="black")
                else:
                    ax[i].plot(x, predicted_data[j-1]["cd"], color=color[j-1])
                ax[i].set_ylabel("$drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)
                ax[i].set_ylim(2.95, 3.25)
            ax[i].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, fontsize=12, ncol=3)
    fig.subplots_adjust(wspace=0.25, top=0.90)
    plt.savefig("".join([path, f"/plots/real_cl_cd_vs_prediction_episode{episode}.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def predict_trajectories(env_model_cl_p: list, env_model_cd: list, path: str, states: pt.Tensor, cd: pt.Tensor,
                         cl: pt.Tensor, actions: pt.Tensor, n_probes: int, n_input_steps: int, min_max: dict,
                         len_trajectory: int = 200) -> dict:
    """
    predict a trajectory based on a given initial state and action using trained environment models for cd, and cl-p

    :param env_model_cl_p: list containing the trained environment model ensemble for cl and p
    :param env_model_cd: list containing the trained environment model ensemble for cd
    :param path: path to the directory where the training is currently running
    :param states: pressure at probe locations sampled from trajectories generated by within CFD used as initial states
    :param cd: cd sampled from trajectories generated by within CFD used as initial states
    :param cl: cl sampled from trajectories generated by within CFD used as initial states
    :param actions: actions sampled from trajectories generated by within CFD used as initial states
    :param n_probes: number of probes places in the flow field
    :param n_input_steps: number as input time steps for the environment models
    :param min_max: the min- / max-values used for scaling the trajectories to the intervall [0, 1]
    :param len_trajectory: length of the trajectory, 1sec CFD = 100 epochs
    :return: the predicted trajectory
    """

    # test model: loop over all test data and predict the trajectories based on given initial state and actions
    # for each model of the ensemble: load the current state dict
    for model in range(len(env_model_cl_p)):
        env_model_cl_p[model].load_state_dict(pt.load(f"{path}/cl_p_model/bestModel_no{model}_val.pt"))
        env_model_cd[model].load_state_dict(pt.load(f"{path}/cd_model/bestModel_no{model}_val.pt"))

    # use batch for prediction, because batch normalization only works for batch size > 1
    # -> at least 2 trajectories required
    batch_size = 2
    shape = (batch_size, len_trajectory)
    traj_cd, traj_cl, traj_actions, traj_p = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape),\
                                             pt.zeros((batch_size, len_trajectory, n_probes))
    for i in range(batch_size):
        traj_cd[i, :n_input_steps] = cd[:n_input_steps]
        traj_cl[i, :n_input_steps] = cl[:n_input_steps]
        traj_actions[i, :n_input_steps] = actions[:n_input_steps]
        traj_p[i, :n_input_steps, :] = states[:n_input_steps, :]

    # loop over the trajectory, each iteration move input window by one time step
    for t in range(len_trajectory - n_input_steps):
        # create the feature (same for both environment models)
        feature = pt.flatten(pt.concat([traj_p[:, t:t + n_input_steps, :],
                                        (traj_cl[:, t:t + n_input_steps]).reshape([batch_size, n_input_steps, 1]),
                                        traj_cd[:, t:t + n_input_steps].reshape([batch_size, n_input_steps, 1]),
                                        (traj_actions[:, t:t + n_input_steps]).reshape([batch_size, n_input_steps, 1])],
                                       dim=2),
                             start_dim=1)

        # randomly choose an environment model to make a prediction
        tmp_cd_model = env_model_cd[pt.randint(low=0, high=len(env_model_cl_p), size=(1, 1)).item()]
        tmp_cl_p_model = env_model_cl_p[pt.randint(low=0, high=len(env_model_cl_p), size=(1, 1)).item()]

        # make prediction for cd
        traj_cd[:, t + n_input_steps] = tmp_cd_model(feature).squeeze().detach()

        # make prediction for probes and cl
        prediction_cl_p = tmp_cl_p_model(feature).squeeze().detach()
        traj_p[:, t + n_input_steps, :] = prediction_cl_p[:, :n_probes]
        traj_cl[:, t + n_input_steps] = prediction_cl_p[:, -1]

        # add (real) action from the MF-training to the traj_actions tensor
        traj_actions[:, t + n_input_steps] = pt.tensor([actions[t + n_input_steps], actions[t + n_input_steps]])

    # re-scale everything for PPO-training and sort into dict
    cl_rescaled = denormalize_data(traj_cl, min_max["cl"])[0, :]
    cd_rescaled = denormalize_data(traj_cd, min_max["cd"])[0, :]

    # all trajectories in batch are the same, so just take the first one
    output = {"states": denormalize_data(traj_p, min_max["states"])[0, :, :], "cl": cl_rescaled, "cd": cd_rescaled,
              "rewards": 3.0 - (cd_rescaled + 0.1 * cl_rescaled.abs())}

    return output


def simulate_ppo_training(load_path: str, wrapper_function, n_episodes: int = 80, len_traj: int = 200,
                          n_models: int = 1, n_states: int = 12, buffer_size: int = 8, n_input_time_steps: int = 30,
                          which_e_pred: list = None, n_layers_cl_p: int = 3, n_layers_cd: int = 5,
                          n_neurons_cl_p: int = 100, n_neurons_cd: int = 50,
                          return_full_buffer: bool = False) -> Tuple[List[dict], List[dict], dict]:
    """
    simulates a PPO-training in order to compare the results of CFD trajectories with the MB-generated trajectories.
    In order obtain comparable results to the full MB-DRL training routine, the complete training process needs to be
    executed, since the models are successively trained throughout the training. In this function, all data comes from
    an MF-training, executed prior running this script, so the policy / actions are taken from the real (CFD)
    environment. In contrast to the MF-training, like in the MB-DRL training routine, here only in every 5th episode the
    env. models are (re-)trained using the current and last CFD episode (the episode 5 episodes ago).

    :param load_path: path to the MF-data
    :param wrapper_function: function of the training routine (which training routine should be executed)
    :param n_episodes: number of episodes to run
    :param len_traj: length of the trajectories
    :param n_models: number of environment models in each ensemble
    :param n_states: number of probes places in the flow field
    :param buffer_size: buffer size
    :param n_input_time_steps: number of input time steps for the environment models
    :param which_e_pred: which episodes should be predicted by the env. models for comparison with CFD
    :param n_layers_cl_p: number of neurons per layer for the cl-p-environment model
    :param n_neurons_cl_p: number of hidden layers for the cl-p-environment model
    :param n_neurons_cd: number of neurons per layer for the cd-environment model
    :param n_layers_cd: number of hidden layers for the cd-environment model
    :param return_full_buffer: flag if 'True': the buffer is filled completely with predicted trajectories,
                               else only the 1st trajectory og the buffer is returned (predicted as well as CFD)
    :return: predicted trajectories, corresponding MF-trajectories and all losses from all env. models
    """
    if which_e_pred is None:
        which_e_pred = [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]

    obs_cfd, predicted_traj, ml_traj, loss = [], [], [], []
    for e in range(n_episodes):
        print(f"Start of episode {e}")

        # every 5th episode (re-)train ensemble of environment models based on CFD data
        if e == 0 or e % 5 == 0:
            # save path of CFD episodes -> models should be trained with all CFD data available
            obs_cfd.append("".join([load_path + f"/observations_{e}.pkl"]))

            # in 1st episode: CFD data is used to train environment models for 1st time
            if e == 0:
                cl_p_models, cd_models, l, obs = wrapper_function(load_path, obs_cfd, len_traj, n_states, buffer_size,
                                                                  n_models, n_input_time_steps,
                                                                  n_layers_cl_p=n_layers_cl_p,
                                                                  n_neurons_cd=n_neurons_cd, n_layers_cd=n_layers_cd,
                                                                  n_neurons_cl_p=n_neurons_cl_p)

            # ever 5th episode: models are loaded and re-trained based on CFD data of the current & last CFD episode
            else:
                cl_p_models, cd_models, l, obs = wrapper_function(load_path, obs_cfd, len_traj, n_states, buffer_size,
                                                                  n_models, n_input_time_steps, load=True,
                                                                  n_layers_cl_p=n_layers_cl_p,
                                                                  n_neurons_cd=n_neurons_cd, n_layers_cd=n_layers_cd,
                                                                  n_neurons_cl_p=n_neurons_cl_p)

            # size(l) = [N_models-1, train_loss_cl_p, train_loss_cd, val_loss_cl_p, val_loss_cd, n_epochs]
            loss.append(l)

        # predict the trajectory based on the actions taken in the CFD environment for selected episodes defined
        if e in which_e_pred:
            cfd_data = pickle.load(open("".join([load_path + f"/observations_{e}.pkl"]), "rb"))
            pred_tmp = []

            for b in range(len(cfd_data)):
                # normalize data, always use the 1st trajectory in obs, since buffer should be >=1
                states, min_max_states = normalize_data(cfd_data[b]["states"])
                cd, min_max_cd = normalize_data(cfd_data[b]["cd"])
                cl, min_max_cl = normalize_data(cfd_data[b]["cl"])
                actions, min_max_actions = normalize_data(cfd_data[b]["actions"])

                # min- / max-values used for normalization
                min_max = {"states": min_max_states, "cl": min_max_cl, "cd": min_max_cd, "actions": min_max_actions}
                pred = predict_trajectories(cl_p_models, cd_models, load_path, states, cd, cl, actions, min_max=min_max,
                                            n_input_steps=n_input_time_steps, len_trajectory=len_traj,
                                            n_probes=n_states)
                pred_tmp.append(pred)

            # depending on the purpose, either return the full buffer, e.g. for network architecture study, or just the
            # 1st trajectory of each chosen episode, e.g. if trajectories should be compared
            if return_full_buffer:
                predicted_traj.append(pred_tmp)
                ml_traj.append(cfd_data)
            else:
                predicted_traj.append(pred_tmp[0])
                ml_traj.append(cfd_data[0])

    if n_models > 1:
        losses = {"train_loss_cl_p": pt.cat([loss[i][:, 0, 0, :] for i in range(len(loss))]),
                  "train_loss_cd": pt.cat([loss[i][:, 0, 1, :] for i in range(len(loss))]),
                  "val_loss_cl_p": pt.cat([loss[i][:, 1, 0, :] for i in range(len(loss))]),
                  "val_loss_cd": pt.cat([loss[i][:, 1, 1, :] for i in range(len(loss))])}
    else:
        losses = {}

    return predicted_traj, ml_traj, losses


def compare_training_methods(settings: dict) -> None:
    """
    executes training routine for the two different training methods and compares the predicted trajectories to the real
    ones from CFD and to each other

    :param settings: setup containing all path etc.
    :return: None
    """
    mb, mf = [], 0
    settings["path_controlled"] = settings["save_path"]
    for routine in range(2):
        if routine == 0:
            res, mf, loss = simulate_ppo_training(settings["main_load_path"] + settings["path_MF_case"],
                                                  original_training_routine, n_models=settings["n_models"],
                                                  which_e_pred=settings["e_trajectory"], len_traj=settings["len_traj"])
        else:
            res, mf, loss = simulate_ppo_training(settings["main_load_path"] + settings["path_MF_case"],
                                                  new_training_routine, n_models=settings["n_models"],
                                                  which_e_pred=settings["e_trajectory"], len_traj=settings["len_traj"])

        mb.append(res)

        if settings["n_models"] > 1:
            # plot train and validation losses
            plot_train_validation_loss(settings, pt.mean(loss["train_loss_cl_p"], dim=0),
                                       pt.mean(loss["val_loss_cl_p"], dim=0), pt.mean(loss["train_loss_cd"], dim=0),
                                       pt.mean(loss["val_loss_cd"], dim=0), pt.std(loss["train_loss_cl_p"], dim=0),
                                       pt.std(loss["val_loss_cl_p"], dim=0), pt.std(loss["train_loss_cd"], dim=0),
                                       pt.std(loss["val_loss_cd"], dim=0), case=routine)

    # plot real (CFD) trajectories vs. the predicted ones
    for e in range(len(mf)):
        pred = [mb[0][e], mb[1][e]]
        plot_cl_cd_vs_prediction(settings["main_load_path"] + settings["save_path"], mf[e], pred, settings["color"],
                                 legend=settings["legend"], episode=settings["e_trajectory"][e])
        plot_trajectories_of_probes(settings["main_load_path"] + settings["save_path"], mf[e], pred, settings["color"],
                                    legend=settings["legend"], episode=settings["e_trajectory"][e])


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/robust_MB_DRL_for_flow_control/",  # top-level directory
        "path_MF_case": r"run/compare_training_approaches/e80_r8_b8_f6_MF/seed1/",
        "save_path": "run/compare_training_approaches/",
        "e_trajectory": [0, 1, 2, 3, 4, 75, 76, 77, 78, 79],        # for which episodes should a trajectory be compared
        "n_probes": 12,                                             # number of probes placed in flow field
        "n_models": 5,                                              # number of environment models in the ensembles
        "len_traj": 200,                                            # number of points in trajectory (MF case)
        "color": ["blue", "red", "green", "darkviolet", "black"],   # colors for the cases, uncontrolled = black
        "legend": ["real (MF)", "predicted (MB, original training)", "predicted (MB, new training)"]
    }

    # ensure reproducibility
    manual_seed(1)

    # create directory for plots
    if not path.exists("".join([setup["main_load_path"], setup["save_path"], "plots"])):
        mkdir("".join([setup["main_load_path"], setup["save_path"], "plots"]))

    # compare MB-generated trajectories using the two different training routines
    compare_training_methods(setup)

    # remove temporary directories created for model training
    rmtree("".join([setup["main_load_path"], setup["path_MF_case"], "/cd_model"]))
    rmtree("".join([setup["main_load_path"], setup["path_MF_case"], "/cl_p_model"]))
