"""
    This script implements a model for correcting the MB-generated trajectories, since the MB-trajectories are in
    general too optimistic due to error propagation. Therefore, a model is trained to account for the difference between
    model-generated and CFD trajectories.

    Note: this script uses a lot of functions already defined in other scripts, but here with slight modifications.
          Due to time constraints, these functions are just copy-pasted into this script for now. If this approach
          stabilizes and / or improves the MB-training this script will be implemented in a more efficient way

"""
import pickle
from typing import Tuple

import torch as pt

from drlfoam.environment.env_model_rotating_cylinder_new_training_routine import normalize_data, train_model, FCModel


def rescale_data(x: pt.Tensor, x_min_max: list) -> pt.Tensor:
    """
    reverse the normalization of the data (same as before, just here to avoid circular import due to bad implementation)

    :param x: normalized data
    :param x_min_max: min- and max-value used for normalizing the data
    :return: de-normalized data as tensor
    """
    # x = (x_max - x_min) * x_norm + x_min
    return (x_min_max[1] - x_min_max[0]) * x + x_min_max[0]


def resort_data(observations: list, len_traj: int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    resorts the trajectories sampled in  CFD

    :param observations: trajectories of the current CFD episode
    :param len_traj: length of the trajectory
    :return: tensors containing cl-, p and cd of the current CFD episode
    """
    # sort the trajectories from all workers, for training the models, it doesn't matter from which episodes the data is
    shape, n_col = (len_traj, len(observations)), 0
    cl, cd, p = pt.zeros(shape), pt.zeros(shape), pt.zeros(shape[0], observations[0]["states"].size()[1], shape[1])
    actions = pt.zeros(shape)

    for observation in range(len(observations)):
        # in case a trajectory has no values in it, drlfoam returns emtpy dict
        if not bool(observations[observation]):
            pass
        # omit failed trajectories in case the trajectory only converged partly
        elif observations[observation]["cd"].size()[0] < len_traj:
            pass
        # for some reason sometimes the trajectories are 1 entry too long, in that case ignore the last value
        elif observations[observation]["cd"].size()[0] > len_traj:
            cl[:, n_col] = observations[observation]["cl"][:len_traj]
            cd[:, n_col] = observations[observation]["cd"][:len_traj]
            actions[:, n_col] = observations[observation]["actions"][:len_traj]
            p[:, :, n_col] = observations[observation]["states"][:, :len_traj]
        else:
            cl[:, n_col] = observations[observation]["cl"]
            cd[:, n_col] = observations[observation]["cd"]
            actions[:, n_col] = observations[observation]["actions"]
            p[:, :, n_col] = observations[observation]["states"]
        n_col += 1

    return cl, cd, p, actions


def create_feature_label(real_traj: list, pred_traj: list, len_traj: int, min_max_vals: dict) -> Tuple[dict, dict]:
    """
    create feature-label pairs in order to train the correction models. Here, the model-generated trajectories are used
    as features and the real (CFD) trajectories are the corresponding labels

    :param real_traj: CFD trajectories of the current episode
    :param pred_traj: predicted trajectories of the environment models based on the actions of the CFD trajectories
    :param len_traj: length of the trajectories
    :param min_max_vals: global min- and max-values used for normalization when training the env. model-ensemble
    :return: dict with the predicted and real trajectories of cd and cl
    """
    # re-sort dict with trajectories to tensors, actions of pred & real are the same here
    cl_real, cd_real, p_real, _ = resort_data(real_traj, len_traj)
    cl_pred, cd_pred, p_pred, a_pred = resort_data(pred_traj, len_traj)

    # scale everything to [0, 1] using the global min- max-values
    cd_pred, _ = normalize_data(cd_pred, min_max_vals["cd"])
    cd_real, _ = normalize_data(cd_real, min_max_vals["cd"])
    cl_pred, _ = normalize_data(cl_pred, min_max_vals["cl"])
    cl_real, _ = normalize_data(cl_real, min_max_vals["cl"])
    p_pred, _ = normalize_data(p_pred, min_max_vals["states"])
    p_real, _ = normalize_data(p_real, min_max_vals["states"])
    a_pred, _ = normalize_data(a_pred, min_max_vals["actions"])

    # stack the actions on the feature tensors (action & MB-traj. used as input, real traj. as output)
    cd_pred = pt.cat([cd_pred, a_pred])
    cl_pred = pt.cat([cl_pred, a_pred])
    p_pred = pt.cat([p_pred, a_pred.unsqueeze(1) * pt.ones(p_pred.size())])

    return {"cd": cd_pred, "cl": cl_pred, "states": p_pred}, {"cd": cd_real, "cl": cl_real, "states": p_real}


def train_correction_models(real: list, predicted: list, load_path: str, buffer_size: int, len_traj: int,
                            min_max_vals: dict) -> Tuple[FCModel, FCModel, FCModel]:
    """
    train the models for correcting the trajectories generated by the environment model-ensemble based on the current
    CFD episode and current env. models

    :param real:CFD trajectories of the current episode
    :param predicted: predicted trajectories of the environment models based on the actions of the CFD trajectories
    :param load_path: path to the current run-directory of the training
    :param buffer_size: buffer size
    :param len_traj: length of the trajectories
    :param min_max_vals: global min- and max-values used for normalization when training the env. model-ensemble
    :return: models for correcting the trajectories of cl-, p and cd
    """
    error_model_cl = FCModel(n_inputs=2*len_traj, n_outputs=len_traj, n_layers=5, n_neurons=50)
    error_model_cd = FCModel(n_inputs=2*len_traj, n_outputs=len_traj, n_layers=5, n_neurons=50)
    error_model_p = FCModel(n_inputs=2*len_traj, n_outputs=len_traj, n_layers=5, n_neurons=150)

    features, labels = create_feature_label(real, predicted, len_traj, min_max_vals)

    # split dataset into train- val (idx for cl and cd the same), depending on how many trajectories are available
    if labels["cd"].size()[1] < 4:
        n_val = 1
    else:
        n_val = int(0.3*buffer_size)
    idx_val = pt.multinomial(features["cd"][0, :], num_samples=n_val)
    idx_train = pt.multinomial(features["cd"][0, :], num_samples=buffer_size - n_val)

    # train models
    _ = train_model(error_model_cd, features["cd"][:, idx_train].T, labels["cd"][:, idx_train].T,
                    features["cd"][:, idx_val].T, labels["cd"][:, idx_val].T, save_name="model_error_cd",
                    save_dir="".join([load_path, "/cd_error_model/"]), epochs=1000, stop=-1e-7)
    _ = train_model(error_model_cl, features["cl"][:, idx_train].T, labels["cl"][:, idx_train].T,
                    features["cl"][:, idx_val].T, labels["cl"][:, idx_val].T, save_name="model_error_cl",
                    save_dir="".join([load_path, "/cl_error_model/"]), epochs=1000, stop=-1e-7)

    # resort the tensor to [batch size, len_traj]
    shape_t = (features["states"][:, :, idx_train].size()[1] * features["states"][:, :, idx_train].size()[2], len_traj*2)
    shape_v = (labels["states"][:, :, idx_val].size()[1] * labels["states"][:, :, idx_val].size()[2], len_traj*2)
    _ = train_model(error_model_p, features["states"][:, :, idx_train].reshape(shape_t),
                    labels["states"][:, :, idx_train].reshape((shape_t[0], len_traj)),
                    features["states"][:, :, idx_val].reshape(shape_v),
                    labels["states"][:, :, idx_val].reshape((shape_v[0], len_traj)),
                    save_name="model_error_p", save_dir="".join([load_path, "/states_error_model/"]), epochs=1000,
                    batch_size=shape_t[0], stop=-1e-7)

    return error_model_cd, error_model_cl, error_model_p


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
    cl_rescaled = rescale_data(traj_cl, min_max["cl"])[0, :]
    cd_rescaled = rescale_data(traj_cd, min_max["cd"])[0, :]

    # all trajectories in batch are the same, so just take the first one
    output = {"states": rescale_data(traj_p, min_max["states"])[0, :, :], "cl": cl_rescaled, "cd": cd_rescaled,
              "rewards": 3.0 - (cd_rescaled + 0.1 * cl_rescaled.abs()),
              "actions": rescale_data(traj_actions, min_max["actions"])[0, :]}

    return output


def predict_traj_for_model_error(cl_p_models: list, cd_models: list, load_path: str, path_cfd: str,
                                 n_input_time_steps: int, n_states: int, len_traj: int,
                                 min_max_vals: dict) -> Tuple[list, list]:
    """
    load the trajectories of the current CFD episode, then used the actions sampled in CFD for generating these
    trajectories with the env. model-ensemble. If there were no uncertainties / errors then the real trajectories and
    the predicted ones would be the same

    :param cl_p_models: model-ensemble containing the models for predicting cl- and p
    :param cd_models: model-ensemble containing the models for predicting cd
    :param load_path: path to the current run-directory of the training
    :param path_cfd: path to the 'observations_*.pkl' file of the current CFD episode
    :param n_input_time_steps: number of time steps used as input for the env. models
    :param n_states: number of probes placed in the flow field
    :param len_traj: length of the trajectories
    :param min_max_vals: global min- and max-values used for normalization when training the env. model-ensemble
    :return:
    """
    # load the trajectories of the current episode
    cfd_data = pickle.load(open(path_cfd, "rb"))
    pred, mf = [], 0

    for b in range(len(cfd_data)):
        # normalize data, always use the 1st trajectory in obs, since buffer should be >=1
        states, _ = normalize_data(cfd_data[b]["states"], min_max_vals["states"])
        cd, _ = normalize_data(cfd_data[b]["cd"], min_max_vals["cd"])
        cl, _ = normalize_data(cfd_data[b]["cl"], min_max_vals["cl"])
        actions, _ = normalize_data(cfd_data[b]["actions"], min_max_vals["actions"])

        # filter out unconverged / broken trajectories before passing into prediction routine
        if cd.size()[0] < len_traj:
            continue
        else:
            pred_tmp = predict_trajectories(cl_p_models, cd_models, load_path, states, cd, cl, actions,
                                            min_max=min_max_vals, n_input_steps=n_input_time_steps,
                                            len_trajectory=len_traj, n_probes=n_states)
            pred.append(pred_tmp)
            mf = cfd_data
    return pred, mf


if __name__ == "__main__":
    pass
