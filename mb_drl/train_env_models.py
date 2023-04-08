"""
contains a class for the environment models, training routines and all required functions for processing the data
regarding the model training.
"""
import os
import sys
import argparse
import torch as pt

from typing import Tuple, Union
from os.path import join, exists

BASE_PATH = os.environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)


class EnvironmentModel(pt.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int, n_neurons: int,
                 activation: callable = pt.nn.functional.leaky_relu):
        """
        implements a fully-connected neural network

        :param n_inputs: N probes * N time steps + (1 action + 1 cl + 1 cd) * N time steps
        :param n_outputs: N probes
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: none
        """
        super(EnvironmentModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.layers = pt.nn.ModuleList()

        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
                self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)


def train_model(model: pt.nn.Module, dataloader_train: pt.utils.data.DataLoader,
                dataloader_val: pt.utils.data.DataLoader, epochs: int = 2500, lr: float = 0.01, stop: float = -1e-7,
                save_name: str = "bestModel", no: int = 0, env: str = "local",
                save_dir: str = "env_model") -> Tuple[list, list] or None:
    """
    train environment model based on sampled trajectories

    :param model: environment model
    :param dataloader_train: dataloader for training
    :param dataloader_val: dataloader for validation
    :param epochs: number of epochs for training
    :param lr: learning rate
    :param stop: if avg. gradient of validation loss reaches this value, the training is aborted
    :param save_dir: path to directory where models should be saved
    :param save_name: name of the model saved, default is number of epoch
    :param no: model number
    :param env: environment, either 'local' or 'slurm', is set in 'run_training.py'
    :return: training and validation loss as list
    """
    if not exists(save_dir):
        # when running multiple trainings parallel, multiple processes try to create this directory at the same time
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            print(f"Info: directory '{save_dir}' was already created by another process.")

    # optimizer settings
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, min_lr=1.0e-4)

    # lists for storing losses
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    training_loss, validation_loss = [], []

    for epoch in range(1, epochs + 1):
        t_loss_tmp, v_loss_tmp = [], []

        # training loop
        model.train()
        for feature, label in dataloader_train:
            optimizer.zero_grad()
            prediction = model(feature).squeeze()
            loss_train = criterion(prediction, label.squeeze())
            loss_train.backward()
            optimizer.step()
            t_loss_tmp.append(loss_train.item())
        training_loss.append(pt.mean(pt.tensor(t_loss_tmp)).to("cpu"))

        # validation loop
        with pt.no_grad():
            for feature, label in dataloader_val:
                prediction = model(feature).squeeze()
                loss_val = criterion(prediction, label.squeeze())
                v_loss_tmp.append(pt.mean(loss_val).item())
        validation_loss.append(pt.mean(pt.tensor(v_loss_tmp)).to("cpu"))

        scheduler.step(metrics=validation_loss[-1])

        if validation_loss[-1] < best_val_loss:
            pt.save(model.state_dict(), join(save_dir, save_name + f"{no}_val.pt"))
            best_val_loss = validation_loss[-1]

        # print some info after every 100 epochs
        if epoch % 100 == 0:
            print(f"epoch {epoch}, avg. training loss = {round(pt.mean(pt.tensor(training_loss[-50:])).item(), 8)}, "
                  f"avg. validation loss = {round(pt.mean(pt.tensor(validation_loss[-50:])).item(), 8)}")

        # check every 50 epochs if model performs well on validation data or validation loss converges
        if epoch % 50 == 0 and epoch >= 150:
            avg_grad_val_loss = (pt.mean(pt.tensor(validation_loss[-5:-1])) -
                                 pt.mean(pt.tensor(validation_loss[-52:-48]))) / 48

            # since loss decreases the gradient is negative, so if it converges or starts increasing, then stop training
            if validation_loss[-1] <= 1e-6 or avg_grad_val_loss >= stop:
                break

    if env == "local":
        return training_loss, validation_loss
    else:
        if save_dir.endswith("cd_model"):
            pt.save(training_loss, join(save_dir, f"loss{no}_train_cd.pt"))
            pt.save(validation_loss, join(save_dir, f"loss{no}_val_cd.pt"))
        else:
            pt.save(training_loss, join(save_dir, f"loss{no}_train_cl_p.pt"))
            pt.save(validation_loss, join(save_dir, f"loss{no}_val_cl_p.pt"))


def train_env_models(path: str, env_model_cl_p: EnvironmentModel, env_model_cd: EnvironmentModel, data_cl_p: list,
                     data_cd: list, epochs: int = 2500, epochs_cd: int = 2500, load: bool = False, env: str = "local",
                     model_no: int = 0) -> list or None:
    """
    initializes two environment models, trains and validates them based on the sampled data from the CFD
    environment. The models are trained and validated using the previous 2 episodes run in the CFD environment

    :param path: path to the directory where the training is currently running
    :param env_model_cl_p: environment model for predicting cl- & p
    :param env_model_cd: environment model for predicting cd
    :param data_cl_p: [dataloader_train, dataloader_val]
    :param data_cd: [dataloader_train, dataloader_val]
    :param epochs: number of epochs for training the cl-p-environment model
    :param epochs_cd: number of epochs for training the cd-environment model
    :param load: flag if models of last episodes should be used as initialization
    :param env: environment, either 'local' or 'slurm', is set in 'run_training.py'
    :param model_no: number of the environment model within the ensemble
    :return: both environment models (cl-p & cd), as well as the corresponding training- and validation losses
    """
    if not exists(path):
        os.mkdir(path)

    # train and validate environment models with CFD data from the previous episode
    print(f"start training the environment model no. {model_no} for cl & p")

    # move model to GPU if available
    device = "cuda" if pt.cuda.is_available() else "cpu"

    # load environment models trained in the previous CFD episode
    if load:
        env_model_cl_p.load_state_dict(pt.load(join(path, "cl_p_model", f"bestModel_no{model_no}_val.pt")))
        env_model_cd.load_state_dict(pt.load(join(path, "cd_model", f"bestModel_no{model_no}_val.pt")))

    # train environment models
    if env == "local":
        train_loss, val_loss = train_model(env_model_cl_p.to(device), dataloader_train=data_cl_p[0],
                                           dataloader_val=data_cl_p[1], save_dir=join(path, "cl_p_model"),
                                           epochs=epochs, save_name=f"bestModel_no", no=model_no, env=env)

        print(f"starting training for cd model no. {model_no}")
        train_loss_cd, val_loss_cd = train_model(env_model_cd.to(device), dataloader_train=data_cd[0],
                                                 dataloader_val=data_cd[1], save_name=f"bestModel_no", no=model_no,
                                                 epochs=epochs_cd, save_dir=join(path, "cd_model"), env=env)

        return [[train_loss, train_loss_cd], [val_loss, val_loss_cd]]

    else:
        train_model(env_model_cl_p.to(device), dataloader_train=data_cl_p[0], dataloader_val=data_cl_p[1],
                    save_dir=join(path, "cl_p_model"), epochs=epochs, save_name=f"bestModel_no", no=model_no, env=env)

        print(f"starting training for cd model no. {model_no}")
        train_model(env_model_cd.to(device), dataloader_train=data_cd[0], dataloader_val=data_cd[1],
                    save_name=f"bestModel_no", no=model_no, epochs=epochs_cd, save_dir=join(path, "cd_model"), env=env)


def execute_model_training_slurm(model_no: int, train_path: str = "examples/run_training/") -> None:
    """
    executes the model training on an HPC cluster using the singularity container

    :param model_no: number of the current environment model
    :param train_path: path to current PPO-training directory
    :return: None
    """
    # cwd = 'drlfoam/drlfoam/environment/', so go back to the training directory
    os.chdir(join("..", "..", "examples"))

    # update path (relative path not working on cluster)
    train_path = join(BASE_PATH, "examples", train_path)

    # initialize each model with different seed value
    pt.manual_seed(model_no)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(model_no)

    settings = pt.load(join(train_path, "settings_model_training.pt"))
    loader_train = pt.load(join(train_path, "loader_train.pt"))
    loader_val = pt.load(join(train_path, "loader_val.pt"))
    loader_train_cd = pt.load(join(train_path, "loader_train_cd.pt"))
    loader_val_cd = pt.load(join(train_path, "loader_val_cd.pt"))
    train_env_models(train_path, settings["env_model_cl_p"], settings["env_model_cd"],
                     data_cl_p=[loader_train[model_no], loader_val[model_no]],
                     data_cd=[loader_train_cd[model_no], loader_val_cd[model_no]],
                     epochs=settings["epochs"], epochs_cd=settings["epochs_cd"], load=settings["load"],
                     model_no=model_no, env="slurm")


if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-m", "--model", required=True, help="number of the current environment model")
    ag.add_argument("-p", "--path", required=True, type=str, help="path to training directory")
    args = ag.parse_args()
    execute_model_training_slurm(int(args.model), args.path)
