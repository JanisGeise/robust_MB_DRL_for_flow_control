"""
    brief:
        - post-processes the result of the environment model trained and tested in 'train_environment_model.py'
        - plots the results

    dependencies:
        - 'train_environment_model.py'

    prerequisites:
        - execution of the "test_training" function in 'run_training.py' in order to generate trajectories within the
          CFD environment (https://github.com/OFDataCommittee/drlfoam)
"""
import torch as pt
import regex as re
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


def denormalize_data(x: pt.Tensor, x_min_max: list) -> pt.Tensor:
    """
    reverse the normalization of the data

    :param x: normalized data
    :param x_min_max: min- and max-value used for normalizing the data
    :return: de-normalized data as tensor
    """
    # x = (x_max - x_min) * x_norm + x_min
    return (x_min_max[1] - x_min_max[0]) * x + x_min_max[0]


def calculate_error_norm(pred_trajectories: Union[dict, list], cl_test: pt.Tensor, cd_test: pt.Tensor,
                         states_test: pt.Tensor, preserve_dim: str = "") -> pt.Tensor:
    """
    calculates the L2- and L1-norm of the error between the real and the predicted trajectories within the test data set

    :param pred_trajectories: predicted trajectories by the environment model
    :param cl_test: cl coefficients of the test data set sampled in the CFD environment
    :param cd_test: cd coefficients of the test data set sampled in the CFD environment
    :param states_test: states at the probe locations of the test data set sampled in the CFD environment
    :param preserve_dim: over which dimension should the norm be calculated, either 'epochs' or 'episodes'; or, if no
                         parameter specified, then norm is calculated over complete input (one value for each parameter
                         and norm for the complete input tensor)
    :return: tensor containing the L2- and L1-norm of the error, the data is stored as
             [[L2-norm states, L2-norm cl, L2-norm cd], [L1-norm states, L1-norm cl, L1-norm cd]]
    """
    if preserve_dim == "epochs":
        l2 = pt.nn.MSELoss(reduction="none")
        l1 = pt.nn.L1Loss(reduction="none")
        losses = [(pt.sum(l2(states_test, pred_trajectories["states"]), dim=1).detach().numpy(),
                   l2(cl_test, pred_trajectories["cl"]).detach().numpy(),
                   l2(cd_test, pred_trajectories["cd"]).detach().numpy()),
                  (pt.sum(l1(states_test, pred_trajectories["states"]), dim=1).detach().numpy(),
                   l1(cl_test, pred_trajectories["cl"]).detach().numpy(),
                   l1(cd_test, pred_trajectories["cd"]).detach().numpy())]
        all_losses = np.array(losses)

    elif preserve_dim == "episodes":
        # calculate MSE (L2) and L1 loss
        l2 = pt.nn.MSELoss()
        l1 = pt.nn.L1Loss()
        all_losses = [(l2(states_test, pred_trajectories["states"]).item(), l2(cl_test, pred_trajectories["cl"]).item(),
                       l2(cd_test, pred_trajectories["cd"]).item()),
                      (l1(states_test, pred_trajectories["states"]).item(), l1(cl_test, pred_trajectories["cl"]).item(),
                       l1(cd_test, pred_trajectories["cd"]).item())]

    else:
        # resort prediction data from list of dict to tensors, so that they have the same shape as the test data
        pred_states, pred_cl, pred_cd = pt.zeros(states_test.size()), pt.zeros(cl_test.size()), pt.zeros(cd_test.size())
        for trajectory in range(len(pred_trajectories)):
            pred_states[:, :, trajectory] = pred_trajectories[trajectory]["states"]
            pred_cl[:, trajectory] = pred_trajectories[trajectory]["cl"]
            pred_cd[:, trajectory] = pred_trajectories[trajectory]["cd"]

        # calculate MSE (L2) and L1 loss
        l2 = pt.nn.MSELoss()
        l1 = pt.nn.L1Loss()
        all_losses = [(l2(states_test, pred_states).item(), l2(cl_test, pred_cl).item(), l2(cd_test, pred_cd).item()),
                      (l1(states_test, pred_states).item(), l1(cl_test, pred_cl).item(), l1(cd_test, pred_cd).item())]

    return pt.tensor(all_losses)


def plot_train_validation_loss(path: str, mse_train: Union[list, pt.Tensor], mse_val: Union[list, pt.Tensor],
                               std_dev_train: Union[list, pt.Tensor] = None, std_dev_val: Union[list, pt.Tensor] = None,
                               episode_wise: bool = False) -> None:
    """
    plots the avg. train- and validation loss and the corresponding std. deviation, if only one training was conducted,
    then only the training- and validation loss is plotted without std. deviation

    :param path: path where the plot should be saved
    :param mse_train: list or tensor containing the (mean) training loss
    :param mse_val: list or tensor containing the (mean) validation loss
    :param std_dev_train: list or tensor containing the (std. deviation) training loss
    :param std_dev_val: list or tensor containing the (std. deviation) validation loss
    :param episode_wise: if the losses should be plotted wrt to the episode number (if = 'True'),
                         if = 'False': loss is plotted wrt epoch number
    :return: None
    """
    plt.plot(range(len(mse_train)), mse_train, color="blue", label="training loss")
    plt.plot(range(len(mse_val)), mse_val, color="red", label="validation loss")
    if std_dev_train is not None:
        plt.fill_between(range(len(mse_val)), mse_val - std_dev_val, mse_val + std_dev_val, color="red", alpha=0.3)
        plt.fill_between(range(len(mse_train)), mse_train - std_dev_train, mse_train + std_dev_train, color="blue",
                         alpha=0.3)

    if episode_wise:
        plt.xlabel("$episode$ $number$", usetex=True, fontsize=13)
        name = "/plots/train_val_loss_normalized_episodewise.png"
    else:
        plt.xlabel("$epoch$ $number$", usetex=True, fontsize=13)
        name = "/plots/train_val_loss_normalized.png"
        plt.yscale("log")
    plt.ylabel("$MSE$ $loss$", usetex=True, fontsize=13)
    plt.legend()
    plt.savefig(path + name, dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_mean_std_error_of_test_data(settings: dict, mean_data: Union[list, pt.Tensor], std_dev: Union[list, pt.Tensor],
                                     norm: str = "l2", parameter: str = "epochs") -> None:
    """
    plots the L1- and L2-norm of the prediction error either along a trajectory (avg. over the whole test data set
    independent of the episode) or wrt to the episode (avg. over all trajectories within each episode) using the
    normalized data

    :param settings: setup containing all the paths etc.
    :param mean_data: tensor containing the mean L2- and L1-norm of the error wrt epochs or episodes;
                      the data is stored as
                      [[L2-norm states, L2-norm cl, L2-norm cd], [L1-norm states, L1-norm cl, L1-norm cd]]
    :param std_dev: tensor containing the std. deviation of L2- and L1-norm of the error wrt epochs or episodes;
                      the data is stored as
                      [[L2-norm states, L2-norm cl, L2-norm cd], [L1-norm states, L1-norm cl, L1-norm cd]]
    :param norm: which norm should be plotted, either 'l1' or 'l2'
    :param parameter: either 'epochs' for avg. over whole test data set, or 'episodes' for plotting wrt to episode no.
    :return: None
    """
    if norm == "l2":
        norm = 0
        ylabel = "$L_2-norm$ $(relative$ $prediction$ $error)$"
    else:
        norm = 1
        ylabel = "$L_1-norm$ $(relative$ $prediction$ $error)$"

    if parameter == "epochs":
        x = range(settings["len_trajectory"])
        xlabel = "$epoch$ $number$"
        # plot mean and std. dev.
        plt.plot(x, mean_data[norm, 0, :], color="red", label="$states$")
        plt.plot(x, mean_data[norm, 1, :], color="blue", label="$c_l$")
        plt.plot(x, mean_data[norm, 2, :], color="green", label="$c_d$")
        plt.fill_between(x, mean_data[norm, 0, :] - std_dev[norm, 0, :], mean_data[norm, 0, :] + std_dev[norm, 0, :],
                         color="red", alpha=0.3)
        plt.fill_between(x, mean_data[norm, 1, :] - std_dev[norm, 1, :], mean_data[norm, 1, :] + std_dev[norm, 1, :],
                         color="blue", alpha=0.3)
        plt.fill_between(x, mean_data[norm, 2, :] - std_dev[norm, 2, :], mean_data[norm, 2, :] + std_dev[norm, 2, :],
                         color="green", alpha=0.3)

    else:
        x = range(mean_data.size()[0])
        xlabel = "$episode$ $number$"
        # plot mean and std. dev.
        plt.plot(x, mean_data[:, norm, 0], color="red", label="$states$")
        plt.plot(x, mean_data[:, norm, 1], color="blue", label="$c_l$")
        plt.plot(x, mean_data[:, norm, 2], color="green", label="$c_d$")
        plt.fill_between(x, mean_data[:, norm, 0] - std_dev[:, norm, 0], mean_data[:, norm, 0] + std_dev[:, norm, 0],
                         color="red", alpha=0.3)
        plt.fill_between(x, mean_data[:, norm, 1] - std_dev[:, norm, 1], mean_data[:, norm, 1] + std_dev[:, norm, 1],
                         color="blue", alpha=0.3)
        plt.fill_between(x, mean_data[:, norm, 2] - std_dev[:, norm, 2], mean_data[:, norm, 2] + std_dev[:, norm, 2],
                         color="green", alpha=0.3)

    plt.legend(loc="upper left", framealpha=1.0, fontsize=10, ncol=3)
    plt.xlabel(xlabel, usetex=True, fontsize=12)
    plt.ylabel(ylabel, usetex=True, fontsize=12)
    plt.savefig(settings["load_path"] + settings["model_dir"] +
                f"/plots/total_prediction_error_L{norm + 1}norm_{parameter}.png", dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_trajectories_of_probes(settings: dict, states_test: dict, predicted_data: Union[list, dict, pt.Tensor],
                                n_probes: int, parameter: str = "epochs", episode_no: int = 0) -> None:
    """
    plots the probe trajectory of the probes, sampled from the CFD environment in comparison to the predicted trajectory
    by the environment model

    :param settings: setup containing all the paths etc.
    :param states_test: states (trajectory of probes) sampled in the CFD environment
    :param predicted_data: states (trajectory of probes) predicted by the environment model
    :param n_probes: total number of probes
    :param parameter: either 'epochs' if one model was trained for whole data set or 'episodes' for plotting wrt to
                      episode no. in case a new model was trained each episode
    :param episode_no: number of episode (only used as save name for plot if parameter = "episodes")
    :return: None
    """
    fig1, ax1 = plt.subplots(nrows=n_probes, ncols=1, figsize=(9, 9), sharex="all", sharey="all")

    if parameter == "epochs":
        for i in range(n_probes):
            ax1[i].plot(range(settings["len_trajectory"]), states_test[:, i], color="black")
            ax1[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=40, usetex=True, fontsize=13)
            ax1[i].plot(range(settings["len_trajectory"]), predicted_data[:, i].detach().numpy(),
                        color="red")
    elif parameter == "episodes":
        for i in range(n_probes):
            ax1[i].plot(range(settings["len_trajectory"]), states_test[:, i], color="black")
            ax1[i].set_ylabel(f"$probe$ ${i + 1}$", rotation="horizontal", labelpad=40, usetex=True, fontsize=13)
            ax1[i].plot(range(settings["len_trajectory"]), predicted_data[:, i].detach().numpy(),
                        color="red")
    fig1.subplots_adjust(hspace=0.75)
    ax1[-1].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    fig1.tight_layout()

    if parameter == "epochs":
        plt.savefig(settings["load_path"] + settings["model_dir"] + "/plots/real_trajectories_vs_prediction.png",
                    dpi=600)
    elif parameter == "episodes":
        plt.savefig(settings["load_path"] + settings["model_dir"] +
                    f"/plots/real_trajectories_vs_prediction_episode{episode_no}.png", dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_cl_cd_vs_prediction(settings: dict, test_data: dict, predicted_data: Union[list, dict], number: int,
                             episode: int = 0) -> None:
    """
    plots the trajectory of cl and cd, sampled from the CFD environment in comparison to the predicted trajectory by the
    environment model

    :param settings: setup containing all the paths etc.
    :param test_data: trajectory of cl and cd sampled in the CFD environment
    :param predicted_data: trajectory of cl and cd predicted by the environment model
    :param number: number of the trajectory within the data set (either within the episode or in total)
    :param episode: number of the episode (only used for the save name of plot)
    :return: None
    """
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    for i in range(2):
        if i == 0:
            ax2[i].plot(range(settings["len_trajectory"]), test_data["cl_test"][:, number], color="black",
                        label="real")
            ax2[i].plot(range(settings["len_trajectory"]), predicted_data[number]["cl"].detach().numpy(),
                        color="red",
                        label="prediction")
            ax2[i].set_ylabel("$lift$ $coefficient$ $\qquad c_l$", usetex=True, fontsize=13)
        else:
            ax2[i].plot(range(settings["len_trajectory"]), test_data["cd_test"][:, number], color="black")
            ax2[i].plot(range(settings["len_trajectory"]), predicted_data[number]["cd"].detach().numpy(),
                        color="red")
            ax2[i].set_ylabel("$drag$ $coefficient$ $\qquad c_d$", usetex=True, fontsize=13)
        ax2[i].set_xlabel("$epoch$ $number$", usetex=True, fontsize=13)
    fig2.suptitle("coefficients - real vs. prediction", usetex=True, fontsize=16)
    fig2.tight_layout()
    fig2.legend(loc="upper right", framealpha=1.0, fontsize=12, ncol=2)
    fig2.subplots_adjust(wspace=0.25)
    if episode != 0:
        plt.savefig(settings["load_path"] + settings["model_dir"] +
                    f"/plots/real_cl_cd_vs_prediction_episode{episode}.png", dpi=600)
    else:
        plt.savefig(settings["load_path"] + settings["model_dir"] + "/plots/real_cl_cd_vs_prediction.png",
                    dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def import_probe_locations(settings: dict) -> None:
    """
    import and plot probe locations of the tested base case

    :param settings: setup containing the import and save path
    :return: None
    """
    pattern = r"\d.\d+ \d.\d+ \d.\d+"
    with open(settings["load_path"] + settings["path_to_probes"] + "p", "r") as f:
        loc = f.readlines()

    # get coordinates of probes, omit appending empty lists and map strings to floats
    coord = [re.findall(pattern, line) for line in loc if re.findall(pattern, line)]
    positions = [list(map(float, i[0].split())) for i in coord]

    plt.figure(num=1, figsize=(5, 5))
    plt.plot(positions[:, 0], positions[:, 1], linestyle="None", marker="o", color="black", label="probes")
    plt.xlabel("$x-position$", usetex=True)
    plt.ylabel("$y-position$", usetex=True)
    plt.legend()
    plt.savefig(settings["load_path"] + settings["model_dir"] + "/plots/probe_positions.png", dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def post_process_results_one_model(settings: dict, predictions: list, cfd_data: dict, train_loss: pt.Tensor,
                                   val_loss: pt.Tensor, plot_probe_loc: bool = False) -> None:
    """
    post-processes the data if only one environment model is used for the complete data set
    ('episode_depending_model = False' in setup)

    :param settings: setup containing all the paths etc.
    :param predictions: contains the predicted trajectories by the environment model
    :param cfd_data: contains the trajectories sampled in the CFD environment
    :param train_loss: list or tensor containing the training loss
    :param val_loss: list or tensor containing the validation loss
    :param plot_probe_loc: if the probe locations should be plotted as well
    :return: None
    """
    # plot training and validation loss
    plot_train_validation_loss(path=settings["load_path"] + settings["model_dir"], mse_train=train_loss,
                               mse_val=val_loss)

    # calculate the mean and std. deviation prediction error along the trajectories for all tested data
    loss = pt.zeros((cfd_data["cl_test"].size()[1], 2, 3, settings["len_trajectory"]))
    for i in range(cfd_data["cl_test"].size()[1]):
        # first calculate MSE (L2) and L1 loss for the normalized data
        loss[i, :, :, :] = calculate_error_norm({"states": predictions[i]["states"], "cl": predictions[i]["cl"],
                                                 "cd": predictions[i]["cd"]}, cfd_data["cl_test"][:, i],
                                                cfd_data["cd_test"][:, i], cfd_data["states_test"][:, :, i],
                                                preserve_dim="epochs")

        # then reverse normalization of output data (= predicted trajectories)
        if settings["normalize"]:
            predictions[i]["cl"] = denormalize_data(predictions[i]["cl"], cfd_data["min_max_cl"])
            predictions[i]["cd"] = denormalize_data(predictions[i]["cd"], cfd_data["min_max_cd"])
            predictions[i]["states"] = denormalize_data(predictions[i]["states"], cfd_data["min_max_states"])

    # de-normalize the test data
    if settings["normalize"]:
        cfd_data["cl_test"] = denormalize_data(cfd_data["cl_test"], cfd_data["min_max_cl"])
        cfd_data["cd_test"] = denormalize_data(cfd_data["cd_test"], cfd_data["min_max_cd"])
        cfd_data["states_test"] = denormalize_data(cfd_data["states_test"], cfd_data["min_max_states"])

    # plot the mean and std. deviation for all test data wrt number of epoch
    plot_mean_std_error_of_test_data(settings, mean_data=pt.mean(loss, dim=0), std_dev=pt.std(loss, dim=0), norm="l1",
                                     parameter="epochs")
    plot_mean_std_error_of_test_data(settings, mean_data=pt.mean(loss, dim=0), std_dev=pt.std(loss, dim=0), norm="l2",
                                     parameter="epochs")

    # plot states of each probe for a random trajectory within the test data set and compare with model prediction
    trajectory_no = pt.randint(low=0, high=cfd_data["actions_test"].size()[1], size=[1, 1]).item()
    plot_trajectories_of_probes(settings, states_test=cfd_data["states_test"][:, :, trajectory_no],
                                predicted_data=predictions[trajectory_no]["states"], parameter="epochs",
                                n_probes=cfd_data["n_probes"])

    # compare real cl and cd values sampled in CFD environment with predicted ones along the trajectory
    plot_cl_cd_vs_prediction(settings, cfd_data, predictions, trajectory_no)

    if plot_probe_loc:
        import_probe_locations(settings)


def post_process_results_episode_wise_model(settings: dict, predictions: list, cfd_data: dict, train_loss: pt.Tensor,
                                            val_loss: pt.Tensor, plot_probe_loc: bool = False) -> None:
    """
    post-processes the data if one environment model is used for each new episode
    ('episode_depending_model = True' in setup)

    :param settings: setup containing all the paths etc.
    :param predictions: contains the predicted trajectories by the environment model
    :param cfd_data: contains the trajectories sampled in the CFD environment
    :param train_loss: list or tensor containing the training loss
    :param val_loss: list or tensor containing the validation loss
    :param plot_probe_loc: if the probe locations should be plotted as well
    :return: None
    """

    # plot training- and validation loss averaged over epochs
    plot_train_validation_loss(path=settings["load_path"] + settings["model_dir"], mse_train=pt.mean(train_loss, dim=0),
                               mse_val=pt.mean(val_loss, dim=0), std_dev_train=pt.std(train_loss, dim=0),
                               std_dev_val=pt.std(val_loss, dim=0), episode_wise=True)
    # plot training- and validation loss averaged over episodes
    plot_train_validation_loss(path=settings["load_path"] + settings["model_dir"], mse_train=pt.mean(train_loss, dim=1),
                               mse_val=pt.mean(val_loss, dim=1), std_dev_train=pt.std(train_loss, dim=1),
                               std_dev_val=pt.std(val_loss, dim=1))

    # compute L2- and L1-norm of the prediction error for each trajectory within each episode
    loss = pt.zeros((cfd_data["actions"].size()[0], cfd_data["actions"].size()[2], 2, 3))
    for e in range(2, cfd_data["actions"].size()[0]):
        for traj in range(cfd_data["actions"].size()[2]):
            # first do the loss calculation for the normalized data
            loss[e, traj, :, :] = calculate_error_norm(predictions[e - 2][traj], cfd_data["cl"][e][:, traj],
                                                       cfd_data["cd"][e][:, traj], cfd_data["states"][e][:, :, traj],
                                                       preserve_dim="episodes")

            # then de-normalize all the predicted trajectories within the current episode, predicted trajectories
            # the predicted episodes start at e-2, because for the 1st 2 episodes there exist no predictions
            # -> divided data contains these first 2 episodes, and therefore starts at idx = e
            predictions[e - 2][traj]["cl"] = denormalize_data(predictions[e - 2][traj]["cl"], cfd_data["min_max_cl"])
            predictions[e - 2][traj]["cd"] = denormalize_data(predictions[e - 2][traj]["cd"], cfd_data["min_max_cd"])
            predictions[e - 2][traj]["states"] = denormalize_data(predictions[e - 2][traj]["states"],
                                                                  cfd_data["min_max_states"])

    # de-normalize the sampled data
    cfd_data["cl"] = denormalize_data(cfd_data["cl"], cfd_data["min_max_cl"])
    cfd_data["cd"] = denormalize_data(cfd_data["cd"], cfd_data["min_max_cd"])
    cfd_data["states"] = denormalize_data(cfd_data["states"], cfd_data["min_max_states"])

    # plot the mean and std. deviation for all test data wrt number of epoch
    plot_mean_std_error_of_test_data(settings, mean_data=pt.mean(loss, dim=1), std_dev=pt.std(loss, dim=1), norm="l1",
                                     parameter="episodes")
    plot_mean_std_error_of_test_data(settings, mean_data=pt.mean(loss, dim=1), std_dev=pt.std(loss, dim=1), norm="l2",
                                     parameter="episodes")

    # compare real cl and cd values sampled in CFD environment with predicted ones along the trajectory for 4 different
    # episodes, also compare the states at the probe locations (pick 4 trajectories distributed over all episodes)
    for episode in np.arange(5, stop=len(predictions)+1, step=int(len(predictions) / 4)):
        trajectory_no = pt.randint(low=0, high=cfd_data["actions"].size()[2], size=[1, 1]).item()

        # plot cl and cd of CFD environment vs. model prediction
        plot_cl_cd_vs_prediction(settings, {"cl_test": cfd_data["cl"][episode, :, :],
                                            "cd_test": cfd_data["cd"][episode, :, :]}, predictions[episode - 2],
                                 number=trajectory_no, episode=episode)

        # plot trajectory of states at probe locations of CFD environment vs. model prediction
        plot_trajectories_of_probes(settings, states_test=cfd_data["states"][episode, :, :, trajectory_no],
                                    predicted_data=predictions[episode - 2][trajectory_no]["states"],
                                    n_probes=cfd_data["n_probes"], episode_no=episode, parameter="episodes")

    if plot_probe_loc:
        import_probe_locations(settings)


if __name__ == "__main__":
    pass
