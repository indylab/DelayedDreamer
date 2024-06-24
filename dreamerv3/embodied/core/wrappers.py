import functools
import time
from collections import deque

import numpy as np

from . import base
from . import space as spacelib


class ObsInBuffer:
  
  def __init__(self,
              env_time: int,
              time_step: dict):
    if 'log_obs_time' in time_step.keys():
      # TODO: we may want to remove ObsInBuffer since env_time is redundant.
      assert env_time == time_step['log_obs_time'], "Something is wrong."
    self._env_time: int = env_time
    self._time_step: dict = time_step
      
  @property
  def env_time(self) -> int:
    return self._env_time
      
  @property
  def time_step(self) -> dict:
    return self._time_step


class DelayedEnv(base.Wrapper):
  """DreamerV3 Env style"""

  def __init__(self, env,
               fixed_delay: int,
               maximum_delay: int):
    super().__init__(env)
    self._env_time: int = 0
    self._agn_time: int = 0
    self._fixed_delay: int = fixed_delay
    self._maximum_delay: int = maximum_delay
    self._obs_buffer: deque[ObsInBuffer] = deque[ObsInBuffer]()
    self._act_buffer: deque[dict] = deque[dict]()
    self._done: bool = False
    self._latest_obs: dict = None
    self._reset()
    
    # Update the observation space
    # Note that it's DreamerV3 style.. May need modifications for other envs skeleton.
    self._obs_space: dict = self.env.obs_space.copy()
    self._obs_space['log_obs_time'] = spacelib.Space(np.int32, (1,), low=0, high=np.inf)
    self._obs_space['log_cur_time'] = spacelib.Space(np.int32, (1,), low=0, high=np.inf)
    self._obs_space['log_is_new'] = spacelib.Space(np.bool_, (1,))

  @functools.cached_property
  def obs_space(self) -> dict:
    return self._obs_space

  # This must not be changed from the environment viewpoint.
  @functools.cached_property
  def act_space(self) -> dict:
    return self.env.act_space

  # This must not be changed from the environment viewpoint.
  @functools.cached_property
  def act_buffer_space(self) -> dict:
    act_buffer_space: dict = self.env.act_space.copy()
    act_buffer_space['log_act_time'] = spacelib.Space(np.int32, (), low=-1, high=np.inf)
    return act_buffer_space
  
  @property
  def env_time(self) -> int:
    return self._env_time
  
  @property
  def agn_time(self) -> int:
    return self._agn_time
  
  @property
  def fixed_delay(self) -> int:
    return self._fixed_delay
  
  @property
  def maximum_delay(self) -> int:
    return self._maximum_delay
  
  @property
  def obs_buffer(self) -> deque[ObsInBuffer]:
    return self._obs_buffer
  
  def _act_wrapper(self, act: dict) -> dict:
    act['action'] = act['action'].reshape((-1))
    act["reset"] = np.asarray(np.bool_(act["reset"]))  # bool -> np.bool_
    act["log_act_time"] = np.int32(self._env_time)
    return act

  def _obs_at_retrieving(self, obs: dict, 
                         is_new: bool) -> dict:
    """Add additional info."""
    obs["log_cur_time"] = np.int32([self._env_time])
    obs["log_is_new"] = np.bool_([is_new])
    
    return obs
  
  def _obs_at_acting(self, obs: dict) -> dict:
    """Add additional info."""
    obs["log_obs_time"] = np.int32([self._env_time])
    
    return obs

  def _reset(self):
    self._done = False
    self._obs_buffer = deque[ObsInBuffer]()
    self._act_buffer = deque[dict]([{k:(np.zeros(v.shape) if k!='log_act_time' else np.int32(-1)) for k, v in self.act_buffer_space.items()}])
    self._env_time = 0
    self._agn_time = 0
    self._latest_obs = None

  def _advance_time(self):
    self._env_time += 1
    self._agn_time = max(self._agn_time, self._env_time - self._genereate_delay())

  def step(self, action):
    return self.env.step(action)
    
  def act(self, action) -> None:
    # TODO: sanity check the logic.
    
    if action['reset']:  # TODO: sanity check.. self._done necessary?
      # TODO: (reference) TimeLimit and ActionRepeat
      
      # step (which calls .reset of the base env under the hood)
      obs = self.env.step(action)  # obs is equivalent to transition (s, r, t, info), time_step
      self._reset()
      obs = self._obs_at_acting(obs=obs)
      
      self._obs_buffer.append(ObsInBuffer(self._env_time, obs))
      # TODO: assert obs['is_first']
      
      return None
    
    else:
      # save action first before advance_time
      self._act_buffer.append(self._act_wrapper(action))
      
      # step
      obs = self.env.step(action)
      self._advance_time()
      obs = self._obs_at_acting(obs)
      
      self._obs_buffer.append(ObsInBuffer(self._env_time, obs))
      self._done = obs['is_last']

      return None

  def get_cur_obs(self) -> dict:
    if len(self._obs_buffer) > 0:
      cur_obs = self._obs_buffer[-1].time_step
      cur_obs_bak = cur_obs.copy()
    else:
      cur_obs_bak = self._latest_obs.copy()
    cur_obs_bak = self._obs_at_retrieving(obs=cur_obs_bak, is_new=True)

    assert cur_obs_bak['log_obs_time'] == cur_obs_bak['log_cur_time']
    return cur_obs_bak

  def get_latest_obs(self) -> dict:
    has_update = False
    if self._check_update():
      self._update_latest_obs()
      has_update = True
    assert self._latest_obs is not None, "get_latest_obs() is called before act()"
    
    latest_obs = self._latest_obs.copy()
    latest_obs = self._obs_at_retrieving(obs=latest_obs, is_new=has_update)

    return latest_obs
  
  def get_action_history(self) -> deque[dict]:
    return self._act_buffer

  def _check_update(self) -> bool:
    return (len(self._obs_buffer) != 0) and (self._obs_buffer[0].env_time <= self._agn_time)

  def _update_latest_obs(self) -> None:
    """Update the latest observation and discard everything before""" 
    update = None
    while (len(self._obs_buffer) != 0) and self._obs_buffer[0].env_time <= self._agn_time:
      update = self._obs_buffer[0].time_step
      if len(self._act_buffer):
        if(self._act_buffer[0]['log_act_time'] < update['log_obs_time'] - 1):
          self._act_buffer.popleft()
      self._obs_buffer.popleft()
    assert update is not None, "No update is found."
    
    # Update
    self._latest_obs = update
  
  def _genereate_delay(self) -> int:
    
    # fixed delay for now
    return self.fixed_delay


class TimeLimit(base.Wrapper):

  def __init__(self, env, duration, reset=True):
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self.env.step(action)
      else:
        action.update(reset=False)
        obs = self.env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
    self._done = obs['is_last']
    return obs


class ActionRepeat(base.Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs


class ClipAction(base.Wrapper):

  def __init__(self, env, key='action', low=-1, high=1):
    super().__init__(env)
    self._key = key
    self._low = low
    self._high = high

  def step(self, action):
    clipped = np.clip(action[self._key], self._low, self._high)
    return self.env.step({**action, self._key: clipped})


class NormalizeAction(base.Wrapper):

  def __init__(self, env, key='action'):
    super().__init__(env)
    self._key = key
    self._space = env.act_space[key]
    self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
    self._low = np.where(self._mask, self._space.low, -1)
    self._high = np.where(self._mask, self._space.high, 1)

  @functools.cached_property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = spacelib.Space(np.float32, self._space.shape, low, high)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self.env.step({**action, self._key: orig})


class OneHotAction(base.Wrapper):

  def __init__(self, env, key='action'):
    super().__init__(env)
    self._count = int(env.act_space[key].high)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._count,)
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(self._sample_action, self._count)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert action[self._key].min() == 0.0, action
      assert action[self._key].max() == 1.0, action
      assert action[self._key].sum() == 1.0, action
    index = np.argmax(action[self._key])
    return self.env.step({**action, self._key: index})

  @staticmethod
  def _sample_action(count):
    index = np.random.randint(0, count)
    action = np.zeros(count, dtype=np.float32)
    action[index] = 1.0
    return action


class ExpandScalars(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._obs_expanded = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if space.shape == () and key != 'reward' and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._obs_expanded.append(key)
      self._obs_space[key] = space
    self._act_expanded = []
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if space.shape == () and not space.discrete:
        space = spacelib.Space(space.dtype, (1,), space.low, space.high)
        self._act_expanded.append(key)
      self._act_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = {
        key: np.squeeze(value, 0) if key in self._act_expanded else value
        for key, value in action.items()}
    obs = self.env.step(action)
    obs = {
        key: np.expand_dims(value, 0) if key in self._obs_expanded else value
        for key, value in obs.items()}
    return obs


class FlattenTwoDimObs(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._keys = []
    self._obs_space = {}
    for key, space in self.env.obs_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._keys.append(key)
      self._obs_space[key] = space

  @functools.cached_property
  def obs_space(self):
    return self._obs_space

  def step(self, action):
    obs = self.env.step(action).copy()
    for key in self._keys:
      obs[key] = obs[key].flatten()
    return obs


class FlattenTwoDimActions(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._origs = {}
    self._act_space = {}
    for key, space in self.env.act_space.items():
      if len(space.shape) == 2:
        space = spacelib.Space(
            space.dtype,
            (int(np.prod(space.shape)),),
            space.low.flatten(),
            space.high.flatten())
        self._origs[key] = space.shape
      self._act_space[key] = space

  @functools.cached_property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = action.copy()
    for key, shape in self._origs.items():
      action[key] = action[key].reshape(shape)
    return self.env.step(action)


class CheckSpaces(base.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    for key, value in action.items():
      self._check(value, self.env.act_space[key], key)
    obs = self.env.step(action)
    for key, value in obs.items():
      self._check(value, self.env.obs_space[key], key)
    return obs

  def _check(self, value, space, key):
    if not isinstance(value, (
        np.ndarray, np.generic, list, tuple, int, float, bool)):
      raise TypeError(f'Invalid type {type(value)} for key {key}.')
    if value in space:
      return
    dtype = np.array(value).dtype
    shape = np.array(value).shape
    lowest, highest = np.min(value), np.max(value)
    raise ValueError(
        f"Value for '{key}' with dtype {dtype}, shape {shape}, "
        f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(base.Wrapper):

  def __init__(self, env, key='action', bins=5):
    super().__init__(env)
    self._dims = np.squeeze(env.act_space[key].shape, 0).item()
    self._values = np.linspace(-1, 1, bins)
    self._key = key

  @functools.cached_property
  def act_space(self):
    shape = (self._dims, len(self._values))
    space = spacelib.Space(np.float32, shape, 0, 1)
    space.sample = functools.partial(
        self._sample_action, self._dims, self._values)
    space._discrete = True
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    if not action['reset']:
      assert (action[self._key].min(-1) == 0.0).all(), action
      assert (action[self._key].max(-1) == 1.0).all(), action
      assert (action[self._key].sum(-1) == 1.0).all(), action
    indices = np.argmax(action[self._key], axis=-1)
    continuous = np.take(self._values, indices)
    return self.env.step({**action, self._key: continuous})

  @staticmethod
  def _sample_action(dims, values):
    indices = np.random.randint(0, len(values), dims)
    action = np.zeros((dims, len(values)), dtype=np.float32)
    action[np.arange(dims), indices] = 1.0
    return action


class ResizeImage(base.Wrapper):

  def __init__(self, env, size=(64, 64)):
    super().__init__(env)
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = spacelib.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage(base.Wrapper):

  def __init__(self, env, key='image'):
    super().__init__(env)
    self._key = key
    self._shape = self.env.render().shape

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    spaces[self._key] = spacelib.Space(np.uint8, self._shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    obs[self._key] = self.env.render()
    return obs


class RestartOnException(base.Wrapper):

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    super().__init__(self._ctor())

  def step(self, action):
    try:
      return self.env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self.env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self.env.step(action)
