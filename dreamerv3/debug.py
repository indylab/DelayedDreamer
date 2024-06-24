from embodied.envs import delayed_env, from_gym
import crafter
import gym
import numpy as np

if __name__ == '__main__':
    
    test_env = 'Alien-v4'
    env = gym.make(test_env)

    # env = crafter.Env()
    fixed_delay = 5
    delayed_env = delayed_env.DelayedEnv(env=env, 
                                         fixed_delay=fixed_delay)
    STEPS = 134

    action_space = env.action_space

    interactions = []
    target_interactions = []

    s = delayed_env.reset()
    target_interactions.append(delayed_env.buffer[-1].transition)
    for i in range(STEPS):
        assert(len(delayed_env.buffer) < fixed_delay + 1)
        a = action_space.sample()
        delayed_env.act(a)
        target_interactions.append(delayed_env.buffer[-1].transition)
        if delayed_env.check_update():
            s, r, done, info = delayed_env.get_update()
            interactions.append((s, r, done, info))
    
    passed = True
    for i in range(len(interactions)):
        s, r, done, info = interactions[i]
        ss, rr, donee, infoo = target_interactions[i]
        if not (s == ss).all() or r != rr or done != donee or info != infoo:
            assert False, "Mismatch"
    