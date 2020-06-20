import numpy as np
import gym
import gpflow
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from utils import rollout, policy
from gpflow import set_trainable
np.random.seed(0)

# Introduces a simple wrapper for the gym environment
# Reduces dimensions, avoids non-smooth parts of the state space that we can't model
# Uses a different number of timesteps for planning and testing
# Introduces priors


class DoublePendWrapper():
    def __init__(self):
        self.env = gym.make('InvertedDoublePendulum-v2').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        a1 = np.arctan2(s[1], s[3])
        a2 = np.arctan2(s[2], s[4])
        s_new = np.hstack([s[0], a1, a2, s[5:-3]])
        return s_new

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        if np.abs(ob[0])> 0.90 or np.abs(ob[-3]) > 0.15 or  np.abs(ob[-2]) > 0.15 or np.abs(ob[-1]) > 0.15:
            done = True
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


if __name__=='__main__':
    SUBS = 1
    bf = 40
    maxiter=10
    state_dim = 6
    control_dim = 1
    max_action=1.0 # actions for these environments are discrete
    target = np.zeros(state_dim)
    weights = 5.0 * np.eye(state_dim)
    weights[0,0] = 1.0
    weights[3,3] = 1.0
    m_init = np.zeros(state_dim)[None, :]
    S_init = 0.005 * np.eye(state_dim)
    T = 40
    J = 5
    N = 12
    T_sim = 130
    restarts=True
    lens = []

    env = DoublePendWrapper()

    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=False)
    for i in range(1,J):
        X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS, render=False)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # cur_rew = 0
        # for t in range(0,len(X_new)):
        #     cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        # print('On this episode reward was ', cur_rew)

        # Update dataset
        X = np.vstack((X, X_new[:T, :]))
        Y = np.vstack((Y, Y_new[:T, :]))
        pilco.mgpr.set_data((X, Y))

        lens.append(len(X_new))
        print(len(X_new))
        if len(X_new) > 120: break
