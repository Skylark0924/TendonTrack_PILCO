import numpy as np
from gpflow import config
from gym import make
float_type = config.default_float()


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        ep_return_lst = []
        ep_length =[]
        pre_timestep = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: {}, Epoch_length: {}".format(ep_return_full, ep_length))
            X.append(np.hstack((x, u)))  # state + action
            Y.append(x_new - x)  # \delta state = new_state - state
            ep_return_sampled += r
            x = x_new
            if done:
                x = env.reset()
                # 记录 task length
                ep_length.append(timestep - pre_timestep)
                pre_timestep = timestep

                ep_return_lst.append(ep_return_full)
                ep_return_full = 0
                ep_return_sampled = 0
            if timestep == timesteps-1:
                ep_return_lst.append(ep_return_full)
        print('---------------------------------------')
        print('[ Return list ]: {}, \n[ Average Return ]: {}'.format(ep_return_lst, np.nanmean(np.array(ep_return_lst))))
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_lst, ep_length


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()
