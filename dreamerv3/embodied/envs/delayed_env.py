from collections import deque
from typing import Tuple
import embodied


class ObsInBuffer:
    def __init__(self, 
                 env_time: int, 
                 transition: Tuple):
        self._env_time = env_time
        self._transition = transition
        
    @property
    def env_time(self) -> int:
        return self._env_time
        
    @property
    def transition(self) -> Tuple:
        return self._transition


class DelayedEnv(embodied.Env):

    def __init__(self, env, 
                 fixed_delay: int):
        self._env = env
        self._env_time: int = 0
        self._agn_time: int = 0
        self._buffer: deque[ObsInBuffer] = deque[ObsInBuffer]()
        self.fixed_delay: int = fixed_delay

    @property
    def env(self):
        return self._env
    
    @property
    def env_time(self) -> int:
        return self._env_time
    
    @property
    def agn_time(self) -> int:
        return self._agn_time
    
    @property
    def buffer(self) -> deque[ObsInBuffer]:
        return self._buffer

    def reset(self):
        self._env_time = 0
        self._agn_time = 0
        self._buffer = deque[ObsInBuffer]()
        
        s = self.env.reset()
        transition = (s, 0, False, None)  # TODO: better initialization? (arbitrary)
        self._buffer.append(ObsInBuffer(self._env_time, transition))
        return s

    def act(self, action) -> None:
        transition = self.env.step(action)
        self._env_time += 1
        self._buffer.append(ObsInBuffer(self._env_time, transition))
        self._agn_time = max(self._agn_time, self._env_time - self.genereate_delay())

    def check_update(self) -> bool:
        return (len(self._buffer) != 0) and (self._buffer[0].env_time <= self._agn_time)

    def get_update(self) -> Tuple:
        # return the latest observation and discard 
        # everything before 
        update = ()
        while (len(self._buffer) != 0) and self._buffer[0].env_time <= self._agn_time:
            update = self._buffer[0].transition
            self._buffer.popleft()
        return update

    def genereate_delay(self) -> int:
        # fixed delay for now
        return self.fixed_delay

