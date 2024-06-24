#!/bin/bash
#SBATCH --time=99-00:00:00
#SBATCH --partition=indylab.p
#SBATCH --nodelist=indy-megatron
#SBATCH --cpus-per-task=16
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

# OBS_TYPE=dmc_proprio
# OBS_TYPE=dmc_vision
OBS_TYPE=mujoco_proprio

# DELAY=2
# DELAY=5
# DELAY=10
# DELAY=15
# DELAY=20
# DELAY=30

# RANDOM_POLICY_SEED=1111
# RANDOM_POLICY_SEED=2222
# RANDOM_POLICY_SEED=3333

DELAYS=(5 10)
# DELAYS=(2 20)
SEEDS=(4444 5555)
TASKS=(mujoco_HalfCheetah-v4 mujoco_Reacher-v4 mujoco_Swimmer-v4 mujoco_HumanoidStandup-v4)

for DELAY in "${DELAYS[@]}"; do
    for RANDOM_POLICY_SEED in "${SEEDS[@]}"; do
        for TASK in "${TASKS[@]}"; do
            srun python dreamerv3/train.py --logdir ./logdir/train/delayed/single_step/memoryless/$(echo ${OBS_TYPE} | sed 's/.*_//')/${TASK}/d=${DELAY}/$(date "+%Y%m%d-%H%M%S")\
                                        --configs ${OBS_TYPE} delayed_training delayed_policy_memoryless\
                                        --run.random_policy_seed ${RANDOM_POLICY_SEED}\
                                        --task ${TASK}\
                                        --delay.delay_length ${DELAY} --delay.maximum_delay ${DELAY}
        done
    done
done