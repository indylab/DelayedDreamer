import re
import os

import embodied
import numpy as np


def eval_only_delay(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    prefix_delay = ''
    if 'delay' in ep.keys():
      delay_num = ep.pop('delay')
      prefix_delay = f'delay{delay_num:0>{2}}_'
    logger.add({'length': length, 'score': score}, prefix=f'{prefix_delay}episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())


  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  print('Start evaluation loop.')

  checkpoint.load(os.path.join(args.from_checkpoint, 'checkpoint.ckpt'), keys=['agent'])

  random_agent = embodied.RandomAgent(env.act_space)
  policy_eval = lambda *params: agent.policy_delayed(*params, mode='eval',
                                                    delayed_actor=args.delayed_actor,
                                                    imagination_mode=args.imagination_mode)

  step.load(0)
  while step < args.steps:
    driver.reset()
    driver.delayed_call(policy_eval, random_agent.policy, episodes=max(len(env), args.eval_eps))    
    # if should_log(step):
    #   logger.add(metrics.result())
    #   # logger.add(timer.stats(), prefix='timer')
    #   logger.write(fps=True)
  logger.write()
