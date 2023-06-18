""" Example training script.
"""
import sys
from time import time
import logging
import argparse

from torch import manual_seed, cuda
from shutil import copytree, rmtree
from os import makedirs, environ, system
from os.path import join, exists

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.environment import RotatingCylinder2D, RotatingPinball2D
from drlfoam.agent import PPOAgent
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig

from drlfoam.environment.env_model_rotating_cylinder import *
from drlfoam.environment.predict_trajectories import fill_buffer_from_models


logging.basicConfig(level=logging.INFO)


SIMULATION_ENVIRONMENTS = {
    "rotatingCylinder2D" : RotatingCylinder2D,
    "rotatingPinball2D" : RotatingPinball2D
}

DEFAULT_CONFIG = {
    "rotatingCylinder2D" : {
        "policy_dict" : {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        },
        "value_dict" : {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        }
    },
    "rotatingPinball2D" : {
        "policy_dict" : {
            "n_layers": 2,
            "n_neurons": 512,
            "activation": pt.nn.functional.relu
        },
        "value_dict" : {
            "n_layers": 2,
            "n_neurons": 512,
            "activation": pt.nn.functional.relu
        },
        "policy_lr": 1.0e-4,
        "value_lr": 1.0e-4
    }
}


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    reward_msg = f"Reward mean/min/max: {sum(rt)/len(rt):2.4f}/{min(rt):2.4f}/{max(rt):2.4f}"
    action_mean_msg = f"Mean action mean/min/max: {sum(at_mean)/len(at_mean):2.4f}/{min(at_mean):2.4f}/{max(at_mean):2.4f}"
    action_std_msg = f"Std. action mean/min/max: {sum(at_std)/len(at_std):2.4f}/{min(at_std):2.4f}/{max(at_std):2.4f}"
    logging.info("\n".join((reward_msg, action_mean_msg, action_std_msg)))


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
    ag.add_argument("-m", "--manualSeed", required=False, default=0, type=int,
                    help="seed value for torch")
    ag.add_argument("-c", "--checkpoint", required=False, default="", type=str,
                    help="Load training state from checkpoint file.")
    ag.add_argument("-s", "--simulation", required=False, default="rotatingCylinder2D", type=str,
                    help="Select the simulation environment.")
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
    checkpoint_file = args.checkpoint
    simulation = args.simulation

    # ensure reproducibility
    manual_seed(args.manualSeed)
    if cuda.is_available():
        cuda.manual_seed_all(args.manualSeed)

    # check if the user-specified finish time is greater than the end time of the base case (required for training)
    check_finish_time(BASE_PATH, end_time, simulation)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    if not simulation in SIMULATION_ENVIRONMENTS.keys():
        msg = (f"Unknown simulation environment {simulation}" +
              "Available options are:\n\n" +
              "\n".join(SIMULATION_ENVIRONMENTS.keys()) + "\n")
        raise ValueError(msg)
    if not exists(join(training_path, "base")):
        copytree(join(BASE_PATH, "openfoam", "test_cases", simulation),
                join(training_path, "base"), dirs_exist_ok=True)
    env = SIMULATION_ENVIRONMENTS[simulation]()
    env.path = join(training_path, "base")

    # if debug active -> add execution of bashrc to Allrun scripts, because otherwise the path to openFOAM is not set
    if hasattr(args, "debug"):
        args.set_openfoam_bashrc(path=env.path)
        env_model = SetupEnvironmentModel(n_input_time_steps=args.n_input_time_steps, path=training_path)
    else:
        env_model = SetupEnvironmentModel(path=training_path)

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=env.mpi_ranks, n_nodes=1, partition="queue-1", time="03:00:00",
            constraint="c5a.24xlarge", modules=["openmpi/4.1.5"], job_name="drl_train",
            commands_pre=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc", "source /fsx/drlfoam_main/setup-env"]
        )
        """
        # for AWS
        config = SlurmConfig(n_tasks=env.mpi_ranks, n_nodes=1, partition="queue-1", time="03:00:00",
                             modules=["openmpi/4.1.5"], constraint = "c5a.24xlarge", job_name="drl_train",
                             commands_pre=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc",
                             "source /fsx/drlfoam/setup-env"], commands=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc",
                             "source /fsx/drlfoam/setup-env"])
        """
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -env.action_bounds, env.action_bounds,
                     **DEFAULT_CONFIG[simulation])

    # load checkpoint if provided
    if checkpoint_file:
        logging.info(f"Loading checkpoint from file {checkpoint_file}")
        agent.load_state(join(training_path, checkpoint_file))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0
        buffer.prepare()

    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()
    env_model.last_cfd = starting_episode

    # begin training
    env_model.start_training = time()
    for e in range(starting_episode, episodes):
        logging.info(f"Start of episode {e}")

        # if only 1 model is used, switch every 4th episode to CFD, else determine switching based on model performance
        if env_model.n_models == 1:
            switch = (e % 4 == 0)
        else:
            # for rotatingPinball: the 1st two episodes need to be in CFD, otherwise the rewards of the following
            # MB-episodes are const. & the policy is not improving
            if simulation == "rotatingPinball2D" and e < 2:
                switch = True
            else:
                # else switch depending on model-performance
                switch = env_model.determine_switching(e)

        if e == starting_episode or switch:
            # save path of current CFD episode
            env_model.append_cfd_obs(e)

            # update n_fills
            if e != starting_episode:
                buffer._n_fills = e

            env_model.start_timer()
            buffer.fill()
            env_model.time_cfd_episode()
            states, actions, rewards = buffer.observations

            # set the correct trajectory length
            env_model.len_traj = actions[0].size()[0]

            # in 1st episode: CFD data is used to train environment models for 1st time
            env_model.start_timer()
            if e == starting_episode:
                model_ensemble, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                          env_model.len_traj, env.n_states,
                                                                          buffer_size, env_model.n_models,
                                                                          n_time_steps=env_model.t_input, env=executer,
                                                                          n_actions=env.n_actions)

            # ever CFD episode: models are loaded and re-trained based on CFD data of the current & last CFD episode
            else:
                model_ensemble, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                          env_model.len_traj, env.n_states,
                                                                          buffer_size, env_model.n_models,
                                                                          load=True, env=executer,
                                                                          n_time_steps=env_model.t_input,
                                                                          n_actions=env.n_actions)
            env_model.time_model_training()

            # save train- and validation losses of the environment models
            env_model.save_losses(e, l)

            # reset buffer, policy loss and set the current episode as last CFD episode
            buffer.reset()
            env_model.reset(e)

        # fill buffer with trajectories generated by the environment models
        else:
            # generate trajectories from initial states using policy from previous episode, fill model buffer with them
            env_model.start_timer()
            predicted_traj, current_policy_loss = fill_buffer_from_models(model_ensemble, e, training_path,
                                                                          observation=obs, n_probes=env.n_states,
                                                                          n_input=env_model.t_input,
                                                                          len_traj=env_model.len_traj,
                                                                          buffer_size=buffer_size, agent=agent,
                                                                          env=executer, seed=args.manualSeed,
                                                                          n_actions=env.n_actions)
            env_model.time_mb_episode()
            env_model.policy_loss.append(current_policy_loss)

            # if len(predicted_traj) < buffer size -> discard trajectories from models and go back to CFD
            if len(predicted_traj) < buffer_size:
                buffer._n_fills = e
                env_model.start_timer()
                buffer.fill()
                env_model.time_cfd_episode()
                states, actions, rewards = buffer.observations
                env_model.append_cfd_obs(e)

                # re-train environment models to avoid failed trajectories in the next episode
                env_model.start_timer()
                model_ensemble, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                          env_model.len_traj, env.n_states,
                                                                          buffer_size, env_model.n_models,
                                                                          n_time_steps=env_model.t_input, load=True)
                env_model.time_model_training()
            else:
                # save the model-generated trajectories
                env_model.save(e, predicted_traj)

                # states, actions and rewards required for PPO-training, they are already re-scaled when generated
                states = [predicted_traj[traj]["states"] for traj in range(buffer_size)]
                actions = [predicted_traj[traj]["actions"] for traj in range(buffer_size)]
                rewards = [predicted_traj[traj]["rewards"] for traj in range(buffer_size)]

        # in case no trajectories in CFD converged, use trajectories of the last CFD episodes to train policy network
        if not actions and e >= 5:
            try:
                n_traj = obs["actions"].size()[1]
                traj_n = pt.randint(0, n_traj, size=(buffer_size,))

                # actions and states stored in obs are scaled to interval [0 ,1], so they need to be re-scaled
                actions = [denormalize_data(obs["actions"][:, t.item()], obs["min_max_actions"]) for t in traj_n]
                states = [denormalize_data(obs["states"][:, :, t], obs["min_max_states"]) for t in traj_n]

                # rewards are not scaled to [0, 1] when loading the data since they are not used for env. models
                rewards = [obs["rewards"][:, t.item()] for t in traj_n]

            # if we don't have any trajectories generated within the last 3 CFD episodes, it doesn't make sense to
            # continue with the training
            except IndexError as e:
                logging.critical(f"[run_training.py]: {e}, could not find any valid trajectories from the last 3 CFD"
                                 f"episodes!\nAborting training.")
                exit(0)

        # continue with original PPO-training routine
        print_statistics(actions, rewards)
        env_model.start_timer()
        agent.update(states, actions, rewards)
        env_model.time_ppo_update()
        agent.save_state(join(training_path, f"checkpoint_{e}.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        if not e == episodes - 1:
            buffer.reset()
    env_model.print_info()
    print(f"Training time (s): {time() - env_model.start_training}")


class RunTrainingInDebugger:
    """
    class for providing arguments when running script in IDE (e.g. for debugging). The ~/.bashrc is not executed when
    not running the training from terminal, therefore the environment variables need to be set manually in the Allrun
    scripts
    """

    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 n_input_time_steps: int = 30, seed: int = 0, timeout: int = 1e15, out_dir: str = "examples/TEST"):
        self.command = ". /usr/lib/openfoam/openfoam2206/etc/bashrc"
        self.output = out_dir
        self.iter = episodes
        self.runners = runners
        self.buffer = buffer
        self.finish = finish
        self.environment = "local"
        self.debug = True
        self.n_input_time_steps = n_input_time_steps
        self.manualSeed = seed
        self.timeout = timeout
        self.checkpoint = ""
        # self.simulation = "rotatingCylinder2D"
        self.simulation = "rotatingPinball2D"

    def set_openfoam_bashrc(self, path: str):
        system(f"sed -i '5i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun.pre")
        system(f"sed -i '4i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun")


if __name__ == "__main__":
    # option for running the training in IDE, e.g. in debugger
    DEBUG = False

    if not DEBUG:
        main(parseArguments())
        exit(0)

    else:
        # for debugging purposes, set environment variables for the current directory
        environ["DRL_BASE"] = "/home/janis/Hiwi_ISM/results_drlfoam_MB/drlfoam/"
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

        # test MB-DRL on local machine for cylinder2D: base case runs until t = 4s
        # d_args = RunTrainingInDebugger(episodes=20, runners=4, buffer=4, finish=5, n_input_time_steps=30, seed=0)

        # for pinball: base case runs until t = 200s
        d_args = RunTrainingInDebugger(episodes=10, runners=2, buffer=2, finish=220, n_input_time_steps=30, seed=0)

        # run PPO training
        main(d_args)

        # clean up afterwards
        for dirs in [d for d in glob(d_args.output + "/copy_*")]:
            rmtree(dirs)
        rmtree(d_args.output + "/base")

        try:
            rmtree(d_args.output + "/env_model")
        except FileNotFoundError:
            print("no directories for environment models found.")
