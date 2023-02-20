"""
    brief:
        - this script conducts a parameter study on how the model architecture wrt number of hidden layers and neurons
          influences the prediction accuracy
        - here the new training routine as implemented in the 'env_model_rotating_cylinder_new_training_routine.py'
          script is used
        - the PPO-training is simulated by using trajectory data of a MF-training in order to accelerate the parameter
          study

    dependencies:
        - 'influence_model_architecture.py' from the 'test_env_models' directory for plotting the L1- and L2-losses
        - 'compare_training_routines.py' for simulating the PPO-training routine of the MB-training
        - 'env_model_rotating_cylinder_new_training_routine.py' from the 'mb_drl' directory for executing the
          model training

    prerequisites:
        - execution of the 'run_training.py' function in the 'test_training' directory in order to conduct a training
          and generate trajectories within the CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import torch as pt
from typing import List
from shutil import rmtree
from os import path, mkdir
from torch import manual_seed

from compare_training_routines import simulate_ppo_training
from test_env_models.influence_model_architecture import plot_losses
from mb_drl.env_model_rotating_cylinder_new_training_routine import wrapper_train_env_model_ensemble as train_routine


def resort_trajectories(trajectories: List[dict]) -> dict:
    """
    resort the trajectories of training in order to simplify the loss calculation

    :param trajectories: list with dicts containing all the trajectories from (simulated) training, each list entry has
                         a dict with all trajectories of this episode stored as lists
    :return: dict with tensors containing all trajectories of all episodes
    """
    out = {"cd": [], "cl": [], "states": []}
    for t in range(len(trajectories)):
        for traj in trajectories[t]:
            for key in out:
                if key in traj:
                    out[key].append(traj[key])

    # convert lists with trajectories to one single tensor per key
    out = {key: pt.stack(value) for (key, value) in zip(out.keys(), out.values())}
    return out


def compute_total_prediction_error(real_traj: List[dict], predicted_traj: List[dict], cd_model: bool) -> pt.Tensor:
    """
    compute the error of the predicted trajectories by the environment model-ensemble compared to the 'real'
    trajectories of the MF-training

    :param real_traj: real trajectories from MF-training
    :param predicted_traj: predicted trajectories from (simulated) MB-training
    :param cd_model: flag if the parameter study should be done for the cd-model ensemble ('True') or cl-p-models
    :return: depending on which model ensemble the loss should be calculated the L1- and L2-losses as:
                [L2(cd), L1(cd)] or [[L2(p), L2(cl)], [L1(p), L1(cl)]]
    """
    # resort the list of dicts with trajectories to one tensor containing all trajectories
    real = resort_trajectories(real_traj)
    pred = resort_trajectories(predicted_traj)

    # calculate MSE (L2) and L1 loss
    l2 = pt.nn.MSELoss()
    l1 = pt.nn.L1Loss()

    if cd_model:
        all_losses = pt.tensor([l2(real["cd"], pred["cd"]).item(), l1(real["cd"], pred["cd"]).item()]).unsqueeze(-1)
    else:
        all_losses = pt.tensor([(l2(real["states"], pred["states"]).item(), l2(real["cl"], pred["cl"]).item()),
                      (l1(real["states"], pred["states"]).item(), l1(real["cl"], pred["cl"]).item())])

    return all_losses


def run_parameter_study(load_path: str, neuron_list: list, layer_list: list, n_episodes: int = 80, len_traj: int = 200,
                        n_models: int = 1, buffer_size: int = 8, run_for_cd_model: bool = False) -> pt.Tensor:
    """
    executes the parameter study for a given list of neurons and layers, computes the prediction losses for each of
    these combinations

    :param load_path: path to the trajectories of the MF-case
    :param neuron_list: list containing all the number of neurons per layer which should be tested
    :param layer_list: list containing all the number of hidden layers which should be tested
    :param n_episodes: number of episodes to run, usually the same as number of episodes run in MF-training
    :param len_traj: length of the trajectory, 1s = 100 points (if default sampling rate is used in MF-training)
    :param n_models: number of environment models in each ensemble
    :param buffer_size: buffer size
    :param run_for_cd_model: flag if parameter study is executed for cl-p-model ensemble or cd-model ensemble
    :return: L1- and L2-losses for cl-p or cd (depending on 'run_for_cd_model') for each neuron-layer combination
    """

    # allocate tensor for storing the L1- and L2 loss of states, cl and cd for each neuron-layer combination
    if run_for_cd_model:
        losses = pt.zeros((len(neuron_list), len(layer_list), 2, 1))
    else:
        losses = pt.zeros((len(neuron_list), len(layer_list), 2, 2))

    # loop over neuron- and hidden layer-list
    for n, neurons in enumerate(neuron_list):
        for l, layers in enumerate(layer_list):
            print(f"staring calculation for network with {neurons} neurons and {layers} layers...")

            if run_for_cd_model:
                predictions, real, _ = simulate_ppo_training(load_path, train_routine, len_traj=len_traj,
                                                             n_models=n_models, buffer_size=buffer_size,
                                                             which_e_pred=list(range(0, n_episodes)),
                                                             n_neurons_cd=neurons, n_episodes=n_episodes,
                                                             n_layers_cd=layers, return_full_buffer=True)
            else:
                predictions, real, _ = simulate_ppo_training(load_path, train_routine, len_traj=len_traj,
                                                             n_models=n_models, buffer_size=buffer_size,
                                                             which_e_pred=list(range(0, n_episodes)),
                                                             n_neurons_cl_p=neurons, n_episodes=n_episodes,
                                                             n_layers_cl_p=layers, return_full_buffer=True)

            # calculate L2- and L1-loss for each neuron-layer combination based on predicted trajectories
            losses[n, l, :, :] = compute_total_prediction_error(real, predictions, cd_model=run_for_cd_model)

            print(f"finished calculation for network with {neurons} neurons and {layers} layers")

    return losses


def wrapper_parameter_study(settings: dict) -> None:
    """
    wrapper for executing the parameter study for both model ensembles and plotting of the results

    :param settings: setup containing all path etc.
    :return: None
    """
    losses = []
    for i in range(2):
        if i == 0:
            loss = run_parameter_study("".join([settings["main_load_path"], settings["path_MF_case"]]),
                                       settings["n_neurons_cl_p"], settings["n_layers_cl_p"], settings["n_episodes"],
                                       settings["len_traj"], settings["n_models"], buffer_size=settings["buffer_size"])

        else:
            loss = run_parameter_study("".join([settings["main_load_path"], settings["path_MF_case"]]),
                                       settings["n_neurons_cd"], settings["n_layers_cd"], settings["n_episodes"],
                                       settings["len_traj"], settings["n_models"], buffer_size=settings["buffer_size"],
                                       run_for_cd_model=True)
        losses.append(loss)

    # concatenate the losses-list to one tensor
    losses = pt.cat(losses, dim=-1)

    # save losses
    pt.save(losses, "".join([settings["main_load_path"], settings["save_path"], "/losses.pt"]))

    # re-organize settings dict in order to be able to use the plotting function
    settings_cl_p = {"load_path": settings["main_load_path"], "model_dir": settings["save_path"],
                     "n_neurons": settings["n_neurons_cl_p"], "n_layers": settings["n_layers_cl_p"],
                     "study_cd_model": False}

    settings_cd = {"load_path": settings["main_load_path"], "model_dir": settings["save_path"],
                   "n_neurons": settings["n_neurons_cd"], "n_layers": settings["n_layers_cd"], "study_cd_model": True}

    # plot the losses vs. network architecture as heatmaps
    plot_losses(settings_cl_p, losses, parameter="states")
    plot_losses(settings_cl_p, losses, parameter="cl")
    plot_losses(settings_cd, losses, parameter="cd")


if __name__ == "__main__":
    # Setup
    setup = {
        "main_load_path": r"/media/janis/Daten/Studienarbeit/robust_MB_DRL_for_flow_control/",  # top-level directory
        "path_MF_case": r"run/influence_model_architecture_opt_train_part3/e80_r8_b8_f6_MF/seed0/",
        "save_path": "run/influence_model_architecture_opt_train_part3/",                      # save path for the plots
        "n_probes": 12,                                       # number of probes placed in flow field
        "buffer_size": 8,                                     # buffer size of the MF-case
        "n_models": 1,                                        # number of environment models in the ensembles
        "n_episodes": 80,                                     # number of episodes run in the MF-training
        "len_traj": 200,                                      # number of points in trajectory (MF case)
        "n_neurons_cl_p": [25, 50],                           # number of neurons per layer for the env model for cl & p
        "n_layers_cl_p": [2, 3, 4],                           # number of hidden layers for the env model for cl & p
        "n_neurons_cd": [25, 50],                             # number of neurons per layer for the env model for cd
        "n_layers_cd": [2, 3, 4],                             # number of hidden layers for the env model for cd
    }

    # ensure reproducibility
    manual_seed(0)

    # create directory for plots
    if not path.exists("".join([setup["main_load_path"], setup["save_path"], "plots"])):
        mkdir("".join([setup["main_load_path"], setup["save_path"], "plots"]))

    # run the parameter study for both the cl-p & cd environment models
    wrapper_parameter_study(setup)

    # remove temporary directories created for model training
    rmtree("".join([setup["main_load_path"], setup["path_MF_case"], "/cd_model"]))
    rmtree("".join([setup["main_load_path"], setup["path_MF_case"], "/cl_p_model"]))
