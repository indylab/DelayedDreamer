#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32GB


TASK=mujoco_cheetah_partial-ratio=0.2
OBS_TYPE=mujoco_proprio

# DELAY=2
DELAY=5
# DELAY=10
# DELAY=15
# DELAY=20
# DELAY=30

# RANDOM_POLICY_SEED=1111
# RANDOM_POLICY_SEED=2222
RANDOM_POLICY_SEED=3333

srun python dreamerv3/train.py --logdir ./logdir/partial_cheetah/extended_mlp/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/d=${DELAY}/$(date "+%Y%m%d-%H%M%S")\
                               --configs ${OBS_TYPE} delayed_training delayed_policy_extended_state\
                               --run.random_policy_seed ${RANDOM_POLICY_SEED}\
                               --task ${TASK}\
                               --delay.delay_length ${DELAY} --delay.maximum_delay ${DELAY}

