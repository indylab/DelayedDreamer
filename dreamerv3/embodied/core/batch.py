import numpy as np

from . import base


class BatchEnv(base.Env):

  def __init__(self, envs, parallel):
    assert all(len(env) == 0 for env in envs)
    assert len(envs) > 0
    self._envs = envs
    self._parallel = parallel
    self._keys = list(self.obs_space.keys())

  @property
  def env_time(self):
    return self._envs[0].env_time

  @property
  def agn_time(self):
    # TODO: may vary depending on envs in multi-envs case with stochastic delay.
    return self._envs[0].agn_time

  @property
  def obs_space(self):
    return self._envs[0].obs_space

  @property
  def act_space(self):
    return self._envs[0].act_space

  def maximum_delay(self, idx):
    return self._envs[idx].maximum_delay
  
  def fixed_delay(self, idx):
    return self._envs[idx].fixed_delay

  def __len__(self):
    return len(self._envs)

  def step(self, action):
    assert all(len(v) == len(self._envs) for v in action.values()), (
        len(self._envs), {k: v.shape for k, v in action.items()})
    obs = []
    for i, env in enumerate(self._envs):
      act = {k: v[i] for k, v in action.items()}
      obs.append(env.step(act))
    if self._parallel:
      obs = [ob() for ob in obs]
    return {k: np.array([ob[k] for ob in obs]) for k in obs[0]}
  
  def act(self, action) -> None:
    "A function dedicated to the Delayed Env."
    assert all(len(v) == len(self._envs) for v in action.values()), (
        len(self._envs), {k: v.shape for k, v in action.items()})
    
    obs = []
    for i, env in enumerate(self._envs):
      act = {k: v[i] for k, v in action.items()}
      obs.append(env.act(act))
    if self._parallel:
      obs = [ob() for ob in obs]
    
    for ob in obs:
      assert ob is None, "ob has to None"
    
    return None

  def get_cur_obs(self):
    obs = []
    for i, env in enumerate(self._envs):
      obs.append(env.get_cur_obs())
    if self._parallel:
      obs = [ob() for ob in obs]
    return {k: np.array([ob[k] for ob in obs]) for k in obs[0]}

  def get_latest_obs(self):
    obs = []
    for i, env in enumerate(self._envs):
      obs.append(env.get_latest_obs())
    if self._parallel:
      obs = [ob() for ob in obs]
    return {k: np.array([ob[k] for ob in obs]) for k in obs[0]}

  def get_action_history(self):
    act_hist = []

    for env in self._envs:
      act_hist.append(env.get_action_history())

    if self._parallel:
      act_hist = [act() for act in act_hist]
    # TODO check_shape # May need to update later. (num_env, buffer_len) (num_env, buffer_len, other_dims)
    # TODO Can we make it faster? triple for loop might slow down.
    if len(act_hist[0]) == 0:
      return {}
    return {k: np.array([([ac[i][k] for i in range(len(ac))]) for ac in act_hist]) for k in act_hist[0][0]}

  def render(self):
    return np.stack([env.render() for env in self._envs])

  def close(self):
    for env in self._envs:
      try:
        env.close()
      except Exception:
        pass
