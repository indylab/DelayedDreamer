#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=indylab.p 
#SBATCH --nodelist=indy-megatron
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=32GB

# TASK=dmc_walker_walk
# TASK=dmc_hopper_hop
# TASK=dmc_cheetah_run
# TASK=dmc_cartpole_balance
# TASK=dmc_acrobot_swingup
# TASK=dmc_cartpole_balance_sparse
# TASK=dmc_cup_catch

# TASK=mujoco_Pendulum-v1 

OBS_TYPE=dmc_proprio
# OBS_TYPE=dmc_vision
# OBS_TYPE=mujoco_proprio

DELAY=5

srun python dreamerv3/train.py --logdir ./logdir/train/delayed/single_step/recon/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/d=${DELAY}/$(date "+%Y%m%d-%H%M%S")\
                               --configs ${OBS_TYPE} delayed_training delayed_policy_recon\
                               --task ${TASK}\
                               --delay.delay_length ${DELAY} --delay.maximum_delay ${DELAY}
