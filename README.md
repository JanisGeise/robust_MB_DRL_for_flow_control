# Studienarbeit
Student thesis project, name of repository will be changed once the exact topic / task is decided

### active flow control using DRL and PPO
In order to run the base simulation in OpenFOAM on a local machine, follow the steps as described in 
[ML in CFD, exercise 1](https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_1.ipynb.ipynb) and
[ML in CFD, exercise 11](https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_10_11.ipynb).
In order to train more efficiently, the *main.py* script can be modified as follows:
- set the buffer size at least to `buffer_size = 8`
- set the number of parallel processes `n_worker` according to the available computational resources
- set the PPO iterations at least to `main_ppo_iteration = 70`
- a seed value can be specified for reproducibility using the *torch.manual_seed()* command