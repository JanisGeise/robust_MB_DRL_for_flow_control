""" Example training script.
"""
import sys
import pickle
import argparse

from glob import glob
from time import time
from os.path import join
from torch import manual_seed
from shutil import copytree, rmtree
from os import makedirs, chdir, environ, system

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import PPOAgent
from drlfoam.environment import RotatingCylinder2D
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig
from drlfoam.environment.env_model_rotating_cylinder import *


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


def parseArguments():
    ag = argparse.ArgumentParser()
    ag.add_argument("-o", "--output", required=False, default="test_training", type=str,
                    help="Where to run the training.")
    ag.add_argument("-e", "--environment", required=False, default="local", type=str,
                    help="Use 'local' for local and 'slurm' for cluster execution.")
    ag.add_argument("-i", "--iter", required=False, default=20, type=int,
                    help="Number of training episodes.")
    ag.add_argument("-r", "--runners", required=False, default=4, type=int,
                    help="Number of runners for parallel execution.")
    ag.add_argument("-b", "--buffer", required=False, default=8, type=int,
                    help="Reply buffer size.")
    ag.add_argument("-f", "--finish", required=False, default=8.0, type=float,
                    help="End time of the simulations.")
    ag.add_argument("-t", "--timeout", required=False, default=1e15, type=int,
                    help="Maximum allowed runtime of a single simulation in seconds.")
    ag.add_argument("-s", "--seed", required=False, default=0, type=int,
                    help="seed value for torch")
    args = ag.parse_args()
    return args


def main(args):
    # settings
    training_path = args.output
    episodes = args.iter
    buffer_size = args.buffer
    n_runners = args.runners
    end_time = args.finish
    executer = args.environment
    timeout = args.timeout

    # ensure reproducibility
    manual_seed(args.seed)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join(BASE_PATH, "openfoam", "test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = join(training_path, "base")

    # if debug active -> add execution of bashrc to Allrun scripts, because otherwise the path to openFOAM is not set
    if hasattr(args, "debug"):
        args.set_openfoam_bashrc(path=env.path)
        n_input_time_steps = args.n_input_time_steps
    else:
        n_input_time_steps = 30

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=2, n_nodes=1, partition="standard", time="00:30:00",
            modules=["singularity/latest", "mpi/openmpi/4.1.1/gcc"]
        )
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # execute Allrun.pre script and set new end_time
    buffer.prepare()
    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # epochs = length(trajectory), assume constant sample rate of 100 Hz (default value)
    n_epochs, obs_cfd = int(100 * (end_time - buffer.base_env.start_time)), []

    # begin training
    start_time = time()
    for e in range(episodes):
        print(f"Start of episode {e}")
        # every 5th episode sample from CFD
        if e == 0 or e % 5 == 0:
            # save path of CFD episodes -> models should be trained with all CFD data available
            obs_cfd.append("".join([training_path + f"/observations_{e}.pkl"]))

            # set episode for save_trajectory() method, because n_fills is now updated only every 5th episode
            if e != 0:
                buffer._n_fills = e

            buffer.fill()
            states, actions, rewards = buffer.observations

            # in 1st episode: CFD data is used to train environment models for 1st time
            if e == 0:
                obs_resorted = split_data(obs_cfd, len_traj=n_epochs, n_probes=env.n_states)
                env_model_cl_p, env_model_cd, losses = train_env_models(training_path, n_input_time_steps, env.n_states,
                                                                        observations=obs_resorted)

            # ever 5th episode: models are loaded and re-trained based on CFD data of the current & last CFD episode
            else:
                obs_resorted = split_data(obs_cfd[-2:], len_traj=n_epochs, n_probes=env.n_states)
                env_model_cl_p, env_model_cd, losses = train_env_models(training_path, n_input_time_steps, env.n_states,
                                                                        observations=obs_resorted, epochs=500,
                                                                        epochs_cd=500, load=True)

            # save train- and validation losses of the environment models, omit losses of the 1st episode
            if e > 0:
                losses = {"train_loss_cl_p": pt.tensor(losses[0][0]), "train_loss_cd": pt.tensor(losses[0][1]),
                          "val_loss_cl_p": pt.tensor(losses[1][0]), "val_loss_cd": pt.tensor(losses[1][1])}
                save_trajectories(training_path, e, losses, name="/env_model_loss_")

            # all observations are saved in obs_resorted, so reset buffer
            buffer.reset()

        # fill buffer with trajectories generated by the environment models
        else:
            # generate trajectories from initial states using policy from previous episode, fill model buffer with them
            predicted_traj = fill_buffer_from_models(env_model_cl_p.eval(), env_model_cd.eval(), e, training_path,
                                                     observation=obs_resorted, n_probes=env.n_states,
                                                     n_input=n_input_time_steps, len_traj=n_epochs,
                                                     buffer_size=buffer_size)

            # save the generated trajectories, for now without model buffer instance
            save_trajectories(training_path, e, predicted_traj)

            # get the states, actions and rewards required for PPO-training
            states = [predicted_traj[traj]["states"] for traj in range(buffer_size)]
            actions = [predicted_traj[traj]["actions"] for traj in range(buffer_size)]
            rewards = [predicted_traj[traj]["rewards"] for traj in range(buffer_size)]

        # continue with original PPO-training routine
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save(join(training_path, f"policy_{e}.pkl"),
                   join(training_path, f"value_{e}.pkl"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        buffer.reset()
    print(f"Training time (s): {time() - start_time}")

    # save training statistics
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f, protocol=pickle.HIGHEST_PROTOCOL)


class RunTrainingInDebugger:
    """
    class for providing arguments when running script in IDE (e.g. for debugging). The ~/.bashrc is not executed when
    not running the training from terminal, therefore the environment variables need to be set manually in the Allrun
    scripts
    """

    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 n_input_time_steps: int = 30, seed: int = 0, timeout: int = 1e15,
                 out_dir: str = "examples/TEST_for_debugging"):
        self.command = ". /usr/lib/openfoam/openfoam2206/etc/bashrc"
        self.output = out_dir
        self.iter = episodes
        self.runners = runners
        self.buffer = buffer
        self.finish = finish
        self.environment = "local"
        self.debug = True
        self.n_input_time_steps = n_input_time_steps
        self.seed = seed
        self.timeout = timeout

    def set_openfoam_bashrc(self, path: str):
        system(f"sed -i '5i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun.pre")
        system(f"sed -i '4i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun")


if __name__ == "__main__":
    # option for running the training in IDE, e.g. in debugger
    DEBUG = True

    if not DEBUG:
        main(parseArguments())
        exit(0)

    else:
        # for debugging purposes, set environment variables for the current directory
        environ["DRL_BASE"] = "/media/janis/Daten/Studienarbeit/drlfoam/"
        environ["DRL_TORCH"] = "".join([environ["DRL_BASE"], "libtorch/"])
        environ["DRL_LIBBIN"] = "".join([environ["DRL_BASE"], "/openfoam/libs/"])
        sys.path.insert(0, environ["DRL_BASE"])
        sys.path.insert(0, environ["DRL_TORCH"])
        sys.path.insert(0, environ["DRL_LIBBIN"])

        # set paths to openfoam
        BASE_PATH = environ.get("DRL_BASE", "")
        sys.path.insert(0, BASE_PATH)
        environ["WM_PROJECT_DIR"] = "/usr/lib/openfoam/openfoam2206"
        sys.path.insert(0, environ["WM_PROJECT_DIR"])
        chdir(BASE_PATH)

        # test MB-DRL
        d_args = RunTrainingInDebugger(episodes=10, runners=2, buffer=2, finish=5, n_input_time_steps=30,
                                       out_dir="examples/TEST")
        assert d_args.finish > 4, "finish time needs to be > 4s, (the first 4sec are uncontrolled)"

        # run PPO training
        main(d_args)

        # clean up afterwards
        for dirs in [d for d in glob(d_args.output + "/copy_*")]:
            rmtree(dirs)
        rmtree(d_args.output + "/base")

        try:
            rmtree(d_args.output + "/cd_model")
            rmtree(d_args.output + "/cl_p_model")
        except FileNotFoundError:
            print("no directories for environment models found.")
