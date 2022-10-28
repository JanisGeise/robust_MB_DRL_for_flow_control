#!/bin/bash
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=drl_train
#SBATCH --ntasks-per-node=4

module load python/3.8.2

# adjust path if necessary
source ~/drlfoam/pydrl/bin/activate
source ~/drlfoam/setup-env --container

# start a training, save output to log_seed(seed).log
# $1 = path (-o), $2 = episodes (-i), $3 = runners (-r), $4 = buffer (-b), $5 = finish (-f), $6 = N seed
python3 run_training.py -o "${1}" -i "${2}" -r "${3}" -b "${4}" -f "${5}" -s "${6}" -e "slurm" &> "log_seed${6}.log"
