"""
    brief:
        - script responsible for handling all the loading, sorting and merging of the data created when running the
          PPO-training for different cases

    dependencies:
        - None

    prerequisites:
        - None
"""
import torch as pt

from glob import glob
from os.path import join


def load_trajectory_data(path: str) -> dict:
    """
    load observations_*.pkl files containing all the data generated during training and sort them into a dict

    :param path: path to directory containing the files
    :return: dict with actions, states, cl, cd. Each parameter contains one tensor with the length of N_episodes, each
             entry has all the trajectories sampled in this episode (cols = N_trajectories, rows = length_trajectories)
    """
    files = sorted(glob(path + "observations_*.pt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    observations = [pt.load(open(file, "rb")) for file in files]
    traj_length, counter, mb_episodes = len(observations[0][0]["actions"]), 0, 0

    # sort the trajectories from all workers wrt the episode
    shape = (len(observations), traj_length, len(observations[0]))
    data = {"n_workers": len(observations[0]), "cl": pt.zeros(shape), "cd": pt.zeros(shape), "actions": pt.zeros(shape),
            "rewards": pt.zeros(shape), "alpha": pt.zeros(shape), "beta": pt.zeros(shape),
            "states": pt.zeros((shape[0], shape[1], observations[0][0]["states"].size()[1], shape[2])),
            "no_e_mb": []}

    # tmp dict for merging the data from all runners within each new episode
    shape = (traj_length, data["n_workers"])
    tmp = {"states": pt.zeros((shape[0], observations[0][0]["states"].size()[1], shape[1])), "cl": pt.zeros(shape),
           "cd": pt.zeros(shape), "actions": pt.zeros(shape), "rewards": pt.zeros(shape), "alpha": pt.zeros(shape),
           "beta": pt.zeros(shape)}

    for episode in range(len(observations)):
        for worker in range(len(observations[episode])):
            # in case a trajectory has no values in it, drlfoam returns emtpy dict
            if not bool(observations[episode][worker]):
                counter += 1
                continue
            # omit failed trajectories in case the trajectory only converged partly
            elif observations[episode][worker]["actions"].size()[0] < traj_length:
                counter += 1
                # print(observations[episode][worker]["actions"].size()[0])
                continue
            # in case there exist more points in one trajectory, just take the first len_traj ones (happens sometimes)
            elif observations[episode][worker]["actions"].size()[0] > traj_length:
                # merge data from all runners for each episode
                for key in tmp:
                    if key == "states":
                        tmp[key][:, :, worker] = observations[episode][worker][key][:traj_length, :]
                    else:
                        tmp[key][:, worker] = observations[episode][worker][key][:traj_length]
            else:
                # merge data from all runners for each episode
                for key in tmp:
                    if key == "states":
                        tmp[key][:, :, worker] = observations[episode][worker][key]
                    else:
                        tmp[key][:, worker] = observations[episode][worker][key]

        # check if trajectories are generated by CFD or by env. models -> if there exist a tag then count this episode
        if "generated_by" in observations[episode][0]:
            if observations[episode][0]["generated_by"] == "env_models":
                mb_episodes += 1

                # save the number of the CFD episodes
                data["no_e_mb"].append(episode)

        # sort in the data wrt to episode
        for key in tmp:
            data[key][episode, :, :] = tmp[key]

    # check how many trajectories failed
    if counter > 0:
        print(f"found {counter} failed trajectories")
    else:
        print("found no invalid trajectories")

    # add ratio (MB / MF) episodes = MB_episodes / (all_episodes - MB_episodes)
    data["MB_MF"] = mb_episodes / (len(observations) - mb_episodes)
    data["MF_episodes"] = len(observations) - mb_episodes

    """ maybe not working due to changes in implementation of MB-training
    
    # import and sort training- and validation losses of the environment models, if MB-DRL was used
    if len(glob(path + "env_model_loss_*.pt")) > 0:
        files = natsorted(glob(path + "env_model_loss_*.pt"))
        losses = [pt.load(open(file, "rb")) for file in files]

        shape = (len(losses), losses[0]["val_loss_cd"].size()[0], losses[0]["val_loss_cd"].size()[-1])
        cd_train_loss, cd_val_loss = pt.zeros(shape), pt.zeros(shape)
        cl_p_train_loss, cl_p_val_loss = pt.zeros(shape), pt.zeros(shape)

        # if early stopping is used -> losses don't have the same shape anymore...
        try:
            for l in range(len(losses)):
                cd_train_loss[l, :, :], cd_val_loss[l, :, :] = losses[l]["train_loss_cd"], losses[l]["train_loss_cl_p"]
                cl_p_train_loss[l, :, :], cl_p_val_loss[l, :, :] = losses[l]["val_loss_cd"], losses[l]["val_loss_cl_p"]
        except RuntimeError:
            pass

        shape = (cd_train_loss.size()[0] * cd_train_loss.size()[1], cd_train_loss.size()[-1])
        data["train_loss_cd"], data["val_loss_cd"] = cd_train_loss.reshape(shape), cd_val_loss.reshape(shape)
        data["train_loss_cl_p"], data["val_loss_cl_p"] = cl_p_train_loss.reshape(shape), cl_p_val_loss.reshape(shape)
    """
    return data


def load_all_data(settings: dict) -> list:
    """
    wrapper function for loading the results of the PPO-training and sorting it wrt episodes

    :param settings: setup containing all paths etc.
    :return: a list containing a dictionary with all the data from each case
    """
    # load the results of the training
    loaded_data = []

    if settings["avg_over_cases"]:
        for c in range(len(settings["case_name"])):
            case_data = []

            # assuming each case directory contains subdirectories with training data ran with different seeds,
            # exclude directories named "plots", readme files, logs etc.
            dirs = [d for d in glob(join(settings["main_load_path"], settings["path_controlled"],
                                         settings["case_name"][c], "seed[0-9]"))]

            for d in dirs:
                case_data.append(load_trajectory_data(d + "/"))

            # merge training results from same case, but different seeds episode-wise
            loaded_data.append(merge_results_for_diff_seeds(case_data, n_seeds=len(case_data)))

    else:
        for c in range(len(settings["case_name"])):
            loaded_data.append(load_trajectory_data(join(settings["main_load_path"], settings["path_controlled"],
                                                    settings["case_name"][c])))
    return loaded_data


def average_results_for_each_case(data: list) -> dict:
    """
    average the loaded results of the training periode wrt the episode for each imported case

    :param data: list containing a dict for each imported case containing all the results of the training
    :return: dict containing the mean actions, cl, cd, alpha, beta and rewards as well as the corresponding standard
             deviation wrt the episode
    """
    # calculate the mean and std. deviation in each episode for each case
    avg_data = {"mean_cl": [], "std_cl": [], "mean_cd": [], "std_cd": [], "mean_actions": [], "std_actions": [],
                "mean_rewards": [], "std_rewards": [], "mean_alpha": [], "std_alpha": [], "mean_beta": [],
                "std_beta": [], "tot_mean_rewards": [], "tot_std_rewards": [], "tot_mean_cd": [], "tot_std_cd": [],
                "tot_mean_cl": [], "tot_std_cl": [], "var_beta_fct": [], "buffer_size": [], "len_traj": [],
                "ratio_MB_MF": [], "MF_episodes": [], "std_beta_fct": [], "e_number_mb": []}
    names, keys, losses = {}, ["cl", "cd", "actions", "rewards", "alpha", "beta"], []

    for case in range(len(data)):
        n_episodes, len_trajectory = data[case]["actions"].size()[0], data[case]["actions"].size()[1]

        for key in keys:
            # reshape data and compute mean wrt to episode
            names[key] = data[case][key].reshape((n_episodes, len_trajectory * data[case]["n_workers"]))
            avg_data[f"mean_" + key].append(pt.mean(names[key], dim=1))
            avg_data[f"std_" + key].append(pt.std(names[key], dim=1))

            # total mean of rewards, cl and cd of complete training for each case
            if key != "alpha" and key != "beta" and key != "actions":
                avg_data[f"tot_mean_" + key].append(pt.mean(data[case][key]))
                avg_data[f"tot_std_" + key].append(pt.std(data[case][key]))

        # compute variance of the (mean) beta-distribution of each episode
        # var = (alpha*beta) / ((alpha + beta)^2 * (alpha+beta+1))
        var = (avg_data["mean_alpha"][case] * avg_data["mean_beta"][case]) / \
              ((avg_data["mean_alpha"][case] + avg_data["mean_beta"][case]) ** 2 *
               (avg_data["mean_alpha"][case] + avg_data["mean_beta"][case] + 1))
        std = (avg_data["std_alpha"][case] * avg_data["std_beta"][case]) / \
              ((avg_data["std_alpha"][case] + avg_data["std_beta"][case]) ** 2 *
               (avg_data["std_alpha"][case] + avg_data["std_beta"][case] + 1))
        avg_data["var_beta_fct"].append(var)
        avg_data["std_beta_fct"].append(std)

        # info about the setup, assuming constant sample rate of 100 Hz
        if "n_seeds" in data[case]:
            avg_data["buffer_size"].append(int(data[case]["n_workers"] / data[case]["n_seeds"]))

        else:
            avg_data["buffer_size"].append(data[case]["n_workers"])
        avg_data["len_traj"].append(int(len_trajectory / 100))

        if "ratio_MB_MF" in data[case]:
            avg_data["ratio_MB_MF"].append(data[case]["ratio_MB_MF"])
            avg_data["MF_episodes"].append(data[case]["MF_episodes"])
            avg_data["e_number_mb"].append(data[case]["MB_episode_no"])

        # if environment models are used, get mean and std. of train- and validation losses vs. epoch
        if "train_loss_cd" in data[case]:
            keys_tmp = 2 * ["train_loss_cl_p", "val_loss_cl_p", "train_loss_cd", "val_loss_cd"]
            tmp_mean = {f"mean_" + k: [] for k in keys_tmp}
            tmp_mean.update({f"std_" + k: [] for k in keys_tmp})
            for idx, k in enumerate(tmp_mean):
                if idx < 4:
                    tmp_mean[k] = pt.mean(data[case][keys_tmp[idx]], dim=0)
                else:
                    tmp_mean[k] = pt.std(data[case][keys_tmp[idx]], dim=0)
            losses.append(tmp_mean)
        else:
            losses.append([])
    avg_data["losses"] = losses

    return avg_data


def merge_results_for_diff_seeds(data: list, n_seeds: int) -> dict:
    """
    merge the trajectories of the PPO-trainings for different seeds episode-wise

    prerequisites: all trainings are done with the same setup (same number of workers etc. but e.g. trainings
    initialized with different seeds)

    :param data: the loaded training data from all cases which should be merged
    :param n_seeds: number of cases
    :return: a dictionary containing the merged data
    """
    n_traj = sum([data[seed]["n_workers"] for seed in range(n_seeds)])
    shape = (data[0]["cd"].size(0), data[0]["cd"].size(1), n_traj)
    states = pt.zeros((data[0]["states"].size(0), data[0]["states"].size(1), data[0]["states"].size(2), n_traj))

    merged_data = {"n_workers": n_traj, "n_seeds": n_seeds, "cl": pt.zeros(shape), "cd": pt.zeros(shape),
                   "actions": pt.zeros(shape), "rewards": pt.zeros(shape), "alpha": pt.zeros(shape),
                   "beta": pt.zeros(shape), "ratio_MB_MF": [], "MF_episodes": [], "MB_episode_no": []}
    keys = ["cl", "cd", "actions", "rewards", "alpha", "beta"]

    for seed in range(n_seeds):
        for k in keys:
            merged_data[k][:, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed][k]
        states[:, :, :, data[seed]["n_workers"] * seed:data[seed]["n_workers"] * (seed + 1)] = data[seed]["states"]

    # sort the states into dict
    merged_data["states"] = states

    # get the ration of MB / MF episodes
    for i in range(len(data)):
        if "MB_MF" in data[i]:
            merged_data["ratio_MB_MF"].append(data[i]["MB_MF"])
            merged_data["MF_episodes"].append(data[i]["MF_episodes"])
            merged_data["MB_episode_no"].append(data[i]["no_e_mb"])

    return merged_data


if __name__ == "__main__":
    pass
