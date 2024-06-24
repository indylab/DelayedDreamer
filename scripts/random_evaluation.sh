#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32GB

# TASK=dmc_walker_walk
# TASK=dmc_hopper_hop
# TASK=dmc_cheetah_run
# TASK=dmc_cartpole_balance
# TASK=dmc_acrobot_swingup
# TASK=dmc_finger_spin

# TASK=mujoco_Pendulum-v1
# TASK=mujoco_HalfCheetah-v4
# TASK=mujoco_Reacher-v4
# TASK=mujoco_Swimmer-v4
# TASK=mujoco_Walker2d-v4
TASK=mujoco_HumanoidStandup-v4

# OBS_TYPE=dmc_proprio
# OBS_TYPE=dmc_vision
OBS_TYPE=mujoco_proprio

srun python dreamerv3/train.py --logdir ./logdir/eval/random/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/$(date "+%Y%m%d-%H%M%S")\
                                --configs ${OBS_TYPE} random_evaluation\
                                --run.steps 5000\
                                --task ${TASK}
