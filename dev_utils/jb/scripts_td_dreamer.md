
mdp-dreamer
```bash
cd /home/jb/git/DelayedDreamer; python dreamerv3/train.py \
--configs dmc_vision mdp_dreamer \
--task dmc_cheetah_run \
--logdir ./logdir/dmc_vision_mdp_dreamer_dmc_cheetah_run$(date "+%Y%m%d-%H%M%S")  --save_replay False
```

dreamer
```bash
cd /home/jb/git/DelayedDreamer; python dreamerv3/train.py \
--configs dmc_vision \
--task dmc_cheetah_run \
--logdir ./logdir/dmc_vision_dmc_cheetah_run$(date "+%Y%m%d-%H%M%S")  --save_replay False
```
