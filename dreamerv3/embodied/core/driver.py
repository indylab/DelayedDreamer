import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    acts, self._state = policy(obs, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def delayed_call(self, policy, policy_random, steps=0, episodes=0, prefill=False):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._delayed_step(policy, policy_random, step, episode, prefill)

  def _delayed_step(self, policy, policy_random, step, episode, prefill=False):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    _ = self._env.act(acts)
    obs = self._env.get_latest_obs()
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    action_history = self._env.get_action_history()

    env_idx = 0
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first and obs['log_is_new'][i]:
          self._eps[i].clear()

    # Assume deterministic delay. Assume same delay for parallel envs.
    if (obs['log_cur_time'][env_idx][0] < self._env.fixed_delay(env_idx)):
      acts, self._state = policy_random(obs, self._state, **self._kwargs)
    else:
      if prefill:
        acts, self._state = policy_random(obs, self._state, **self._kwargs)
      else:
        acts, self._state = policy(obs, action_history, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}

    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}

    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    if obs['log_is_new'].any(): # to avoid storing a state multiple times
      for i in range(len(self._env)):
        if obs['log_cur_time'][env_idx][0] == 0:
          # special case for the first observation: action_history contains only a_{-1}
          # dont store this in replay buffer
          next_actions = {k + '_next': v[i][1:] for k, v in action_history.items()}
          d = next_actions[list(next_actions.keys())[0]].shape[0]
          for k, v in next_actions.items():
            pds = self._env.maximum_delay(i) - v.shape[0]
            assert len(v.shape) in [1, 2]
            if len(v.shape) == 1: # forst action['reset']
              next_actions[k] = np.pad(v, pad_width=(0, pds), mode='constant')
            else: # for action['action']
              next_actions[k] = np.pad(v, pad_width=((0, pds), (0, 0)), mode='constant')
          trn = {**{k: v[i] for k, v in obs.items()}, **{k: v[i] for k, v in acts.items()}, **next_actions, 'delay': d}
          [self._eps[i][k].append(v) for k, v in trn.items()]
          step += 1
        else:
          aligned_acts = {k: v[i][1] for k, v in action_history.items()} # a_t
          next_actions = {k + '_next': v[i][1:] for k, v in action_history.items()} # from a_t to a_t+d-1
          # padding next_actions with zeros
          d = next_actions[list(next_actions.keys())[0]].shape[0]
          for k, v in next_actions.items():
            pds = self._env.maximum_delay(i) - v.shape[0]
            assert len(v.shape) in [1, 2]
            if len(v.shape) == 1: # for action['reset']
              next_actions[k] = np.pad(v, pad_width=(0, pds), mode='constant')
            else: # for action['action']
              next_actions[k] = np.pad(v, pad_width=((0, pds), (0, 0)), mode='constant')
          obs_i = {k: v[i] for k, v in obs.items()}
          trn = {**obs_i, **aligned_acts, **next_actions, 'delay': d}
          [self._eps[i][k].append(v) for k, v in trn.items()]
          [fn(trn, i, **self._kwargs) for fn in self._on_steps]  # TODO: check if episode is aligned correctly
          step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          ep['delay'] = self._env.fixed_delay(i)
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
