#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=indylab.p 
#SBATCH --nodelist=indy-megatron
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32GB

# TASK=dmc_walker_walk
# TASK=dmc_hopper_hop
# TASK=dmc_cheetah_run
TASK=dmc_cartpole_balance

# TASK=dmc_acrobot_swingup
# TASK=dmc_cartpole_balance_sparse
# TASK=dmc_cup_catch


OBS_TYPE=dmc_proprio
# OBS_TYPE=dmc_vision

# EVAL_EVERY=15000
# EVAL_EVERY=30000

FROM_CHECKPOINT=./logdir/train/standard/$(echo ${OBS_TYPE} | sed 's/.*_//')/_${TASK}

# srun python dreamerv3/train.py --logdir ./logdir/trian/standard/$(echo ${OBS_TYPE} | sed 's/.*_//')/$(date "+%Y%m%d-%H%M%S")/ --configs ${OBS_TYPE} save_every --run.script train_eval --run.eval_every ${EVAL_EVERY} --task ${TASK}

# srun python dreamerv3/train.py --logdir ./logdir/eval/imagine_recon/$(echo ${OBS_TYPE} | sed 's/.*_//')/$(date "+%Y%m%d-%H%M%S") --configs ${OBS_TYPE} delayed --run.script eval_only_delay --run.steps 5e3 --task ${TASK} --run.from_checkpoint ./logdir/imagine_recon/$(echo ${OBS_TYPE} | sed 's/.*_//')/_${TASK} --delay.delay_eval_version recon --imagine_mode recon

srun python dreamerv3/train.py --logdir ./logdir/eval/standard/$(echo ${OBS_TYPE} | sed 's/.*_//')/$(date "+%Y%m%d-%H%M%S") --configs ${OBS_TYPE} delayed --run.script eval_only_delay --run.steps 5e3 --task ${TASK} --run.from_checkpoint ${FROM_CHECKPOINT} --delay.delay_eval_version recon