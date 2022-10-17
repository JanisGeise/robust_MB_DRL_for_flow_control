#!/bin/bash
# syntax: ./submit_jobs.sh $1 = episodes (-i), $2 = runners (-r), $3 = buffer (-b), $4 = finish (-f)
# example for executing a training with:
#   - 80 episodes
#   - 10 runners
#   - buffer size of 10
#   - finish time of 8sec
# on the cluster: $ ./submit_jobs.sh 80 10 10 8

# run dir is always the same throughout the parameter study
run_dir="e${1}_r${2}_b${3}_f${4}"
mkdir $run_dir

# run each combination for 3 different seeds
for (( seed=0; seed<3; seed++ )); do
    sbatch run_job.sh "${run_dir}/seed${seed}" "${1}" "${2}" "${3}" "${4}" "${seed}"
done
