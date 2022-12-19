# Robust model-based deep reinforcement learning for flow control

## Overview
This student thesis project aims to implement a model-based deep reinforcement learning algorithm for controlling the
flow past a cylinder. Therefore, the [drlfoam](https://github.com/OFDataCommittee/drlfoam) repository, which already 
provides a model-free version is used as a starting point.

This project is a continuation of the work done by [Darshan Thummar](https://github.com/darshan315/flow_past_cylinder_by_DRL) and
[Fabian Gabriel](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL), a first attempt to use a model-based
approach in order to accelerate the training process was implemented by [Eric Schulze](https://github.com/ErikSchulze1796/Active_flow_control_past_cylinder_using_DRL).

## Getting started
### General information
An overview of this repository and information on how to choose parameters for training can be found in the
[overview notebook](https://github.com/JanisGeise/robust_MB_DRL_for_flow_control/blob/main/overview.ipynb). This 
repository only contains all altered and added scripts of *drlfoam* in order to modify the MF-DRL algorithm towards an
MB-version. These scripts can e.g. be downloaded and pasted into an existing (local) *drlfoam* version. Alternatively, a
completed MB-version of *drlfoam* can be found [here](https://github.com/JanisGeise/drlfoam), which is forked from the
[original drlfoam](https://github.com/OFDataCommittee/drlfoam) repository.

### Running a training on a local machine
refer to [drlfoam](https://github.com/OFDataCommittee/drlfoam) for comprehensive guide, here just a brief overview...

additional requirements for running parameter studies etc. can be found in the [requirements.txt](https://github.com/JanisGeise/robust_MB_DRL_for_flow_control/blob/main/requirements.txt)

### Running a training on an HPC cluster
refer to [drlfoam](https://github.com/OFDataCommittee/drlfoam) for comprehensive guide, here just a brief overview...

Examples of shell-scripts for submitting jobs on an HPC cluster (here for the [Phoenix](https://www.tu-braunschweig.de/it/dienste/21/phoenix)
cluster of TU Braunschweig) can be found in [run_job.sh](https://github.com/JanisGeise/robust_MB_DRL_for_flow_control/blob/main/run_job.sh)
and [submit_jobs.sh](https://github.com/JanisGeise/robust_MB_DRL_for_flow_control/blob/main/submit_jobs.sh).

## Troubleshooting
In case something is not working as expected or if you find any bugs, please feel free to open up a new
[issue](https://github.com/JanisGeise/robust_MB_DRL_for_flow_control/issues).

## Report

## References
- the original [drlfoam repository](https://github.com/OFDataCommittee/drlfoam), currently maintained by
  [Andre Weiner](https://github.com/AndreWeiner)
- implementation of the (model-free) PPO algorithm for active flow control:
  * **Thummar, Darshan**. *Active flow control in simulations of fluid flows based on deep reinforcement learning*,
  https://doi.org/10.5281/zenodo.4897961 (May, 2021).
  * **Gabriel, Fabian**. *Aktive Regelung einer Zylinderumströmung bei variierender Reynoldszahl durch bestärkendes Lernen*,
  https://doi.org/10.5281/zenodo.5634050 (October, 2021).

- first attempt of implementing a model-based version for accelerating the training process:
  * **Schulze, Eric**. *Model-based Reinforcement Learning for Accelerated Learning From CFD Simulations*,
  https://doi.org/10.5281/zenodo.6375575 (March, 2022).
