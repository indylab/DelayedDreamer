#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=ava_s.p 
#SBATCH --nodelist=ava-s3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32GB

# TASK=dmc_walker_walk
# TASK=dmc_hopper_hop
# TASK=dmc_cheetah_run
# TASK=dmc_cartpole_balance
# TASK=dmc_acrobot_swingup
# TASK=dmc_finger_spin

# TASK=mujoco_HalfCheetah-v4
TASK=mujoco_Reacher-v4
# TASK=mujoco_Swimmer-v4
# TASK=mujoco_HumanoidStandup-v4

# OBS_TYPE=dmc_proprio
# OBS_TYPE=dmc_vision
OBS_TYPE=mujoco_proprio

CUDA_VISIBLE_DEVICES=1 python dreamerv3/train.py --logdir ./logdir/train/standard/single_step/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/$(date "+%Y%m%d-%H%M%S")\
                                    --configs ${OBS_TYPE}\
                                    --run.script train_eval\
                                    --task ${TASK}

# TASKS=(dmc_walker_walk dmc_hopper_hop dmc_cheetah_run dmc_cartpole_balance dmc_acrobot_swingup dmc_finger_spin)

# for RANDOM_POLICY_SEED in "${SEEDS[@]}"; do
#     for TASK in "${TASKS[@]}"; do
#         srun python dreamerv3/train.py --logdir ./logdir/train/standard/single_step/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/$(date "+%Y%m%d-%H%M%S")\
#                                     --configs ${OBS_TYPE}\
#                                     --run.script train_eval\
#                                     --task ${TASK}
#     done
# done



