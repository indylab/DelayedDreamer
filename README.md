# DelayedDreamer

DelayedDreamer is a modified version of [DreamerV3](https://github.com/danijar/dreamerv3) implemented in JAX and designed to address observation delays in Partially Observable Markov Decision Processes (POMDP) environments. This repository includes code for training and evaluating agents in environments with delayed observations.

## Features

- Including a wrapper for environments to include fixed observation delays.
- Supports different strategies for addressing delays including latent state imagination, extended state, and memoryless approaches.
- Delayed training and evaluation.


## Environment setup

Create a conda environment using the provided `environment.yml` file:

```sh
conda env create -n ddreamer -f environment.yml
```

# Training with delays

To train DelayedDreamer, choose a policy from delayed_policy_latent, delayed_policy_memoryless, or delayed_policy_extended_state:

```sh
python dreamerv3/train.py --logdir ./logdir/$(date "+%Y%m%d-%H%M%S")\
                          --task dmc_walker_walk\
                          --configs proprio delayed_training delayed_policy_latent\
                          --delay.delay_length 5 --delay.maximum_delay 5
```


# Evaluation with delays


To evaluate a trained agent on the server for dmc_proprio:
```sh
python dreamerv3/train.py --logdir ./logdir/$(date "+%Y%m%d-%H%M%S")\
                                        --configs proprio delayed_evaluation delayed_policy_latent\
                                        --run.steps 5000\
                                        --run.from_checkpoint <path_to_checkpoint_directory>\
                                        --task dmc_walker_walk\
                                        --delay.delay_length 5 --delay.maximum_delay 5
```


## Notes

- For --run.from_checkpoint, provide the folder name, not the filename.
- When evaluating a trained agent, use the same policy as during training.
- An undelayed agent can be evaluated in a delayed environment using delayed_policy_latent, referred to as Agnostic in the paper.
- The current version of the code supports only fixed delays, meaning delay_length and maximum_delay should be the same.


## Citation

If you find this code useful, please reference [DreamerV3](https://github.com/danijar/dreamerv3) and our paper:

```
@article{karamzade2024reinforcement,
  title={Reinforcement learning from delayed observations via world models},
  author={Karamzade, Armin and Kim, Kyungmin and Kalsi, Montek and Fox, Roy},
  journal={arXiv preprint arXiv:2403.12309},
  year={2024}
}
```

## Acknowledgements

We would like to thank the authors of [DreamerV3](https://github.com/danijar/dreamerv3) for sharing their work.