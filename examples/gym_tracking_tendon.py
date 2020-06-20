import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import os
import numpy as np
import json
import math
import shutil
import time
from gym.spaces import Discrete, Box
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gpflow import set_trainable
import matplotlib.pyplot as plt
import cv2
import pdb

np.random.seed(0)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 隐藏warning
# sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
# tf.Device(CUDA_VISIBLE_DEVICES=-1)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from utils import rollout, policy

from mujoco_model.continuum_env import *

TASK_HORIZON = 300
GLOBAL_IMAGE_LOG_DIR = '/home/lab/Github/PILCO/log/global_image_log/'  #
VIDEO_LOG_ROOT = '/home/lab/Github/PILCO/log/video_log/'  #

RENDER = False


class TendonGymEnv(MujocoEnv):
    def __init__(self, config=None):
        self.config = config
        self.env = create_env('Goal_LfD', None)
        self.reward = 0
        self.observation_space = Box(-1000 * np.ones(7), 1000 * np.ones(7), dtype='float64')
        self.action_space = Box(-1 * np.ones(4), 1 * np.ones(4), dtype='float64')
        self.prev_state = None

        self.accum_actions_coeff = 0.1

        self.image_num = 0
        self.timestep = 0
        self.reset_count = 0

        self.dis_rew_factor = 0.5  # TODO: Need to tune
        self.info = {'flg': 0, 'obs': 0, }
        self.accum_actions = np.zeros(4)

        timestamp = time.time()
        print(timestamp)
        self.VIDEO_LOG_DIR = VIDEO_LOG_ROOT + str(timestamp) + '/'
        shutil.rmtree(GLOBAL_IMAGE_LOG_DIR)
        os.mkdir(GLOBAL_IMAGE_LOG_DIR)
        # shutil.rmtree(self.VIDEO_LOG_DIR)
        os.mkdir(self.VIDEO_LOG_DIR)

        self.reset()

    def step(self, action):
        assert action.shape == self.accum_actions.shape
        self.accum_actions += action * self.accum_actions_coeff
        # Interaction
        cur_state, cur_images = self.env.step(action)
        cur_state = np.array(cur_state)
        # get reward
        reward = np.array(self.get_reward(cur_state))

        # Save Images and Videos or not
        if RENDER:
            global_fig = self.env.sim.render(width=540, height=720, camera_name='global_camera')

            alpha = 1.0
            beta = 10
            images_1 = np.uint8(np.clip((alpha * cur_images + beta), 0, 255))
            images_2 = np.uint8(np.clip((alpha * global_fig + beta), 0, 255))
            cv2.imwrite(GLOBAL_IMAGE_LOG_DIR + "{}.jpg".format(self.image_num), global_fig)
            self.video.write(cv2.cvtColor(np.concatenate((images_1, images_2), axis=1), cv2.COLOR_BGR2RGB))

        self.timestep += 1
        self.image_num += 1

        self.info = {'flg': False, 'obs': 0, }

        # --------------------done judgement------------------
        if None in cur_state:
            cur_state = self.prev_state  # np.array([335, 335, 0])
            reward -= 10
            done = True

            if RENDER:
                self.video.release()
        elif abs(cur_state[0]) >= 300 * self.env.XY_coeff or abs(cur_state[1]) >= 300 * self.env.XY_coeff:
            # cur_state = np.array([500, 500, 0])
            reward -= 10
            done = True

            if RENDER:
                self.video.release()
        elif abs(cur_state[0]) <= 20 * self.env.XY_coeff and abs(cur_state[1]) <= 20 * self.env.XY_coeff:
            reward += 10
            done = False
            # update target with new position
            stat, [next_state, _] = self.env.update_obj()
            print('state:{}, nextobs:{}'.format(stat, next_state))
            # Save the update infos into info_dict
            self.info['flg'] = True
            self.info['obs'] = np.array(next_state)
            # 如果到达三阶样条终点
            if stat:
                reward += 100
                done = True
                if RENDER:
                    self.video.release()
        else:
            done = False
        if self.timestep >= TASK_HORIZON:
            done = True
            if RENDER:
                self.video.release()
        # ------------------------------------------------------

        # If target is updated, change the prev_state into new target so that
        # we can get a normal reward in next interaction.
        # TODO: how can I amend the obs in replay buffer?
        if self.info['flg']:
            import pdb
            self.prev_state = self.info['obs']
            # pdb.set_trace()
            # 更新的点超出范围
            if None in self.info['obs']:
                reward -= 10
                done = True
                if RENDER:
                    self.video.release()

        cur_state = np.concatenate((cur_state, self.accum_actions))

        return cur_state, reward, done, self.info  # return next_state and reward

    '''
    check right: |s_0|-|s_0'|+|s_1|-|s_1'|+|h-\hat{h}|
    '''

    def get_reward(self, cur_state):
        # reward function
        if None in cur_state or None in self.prev_state:
            reward = -100
        else:
            try:
                reward = (abs(self.prev_state[0]) - abs(cur_state[0])) + (abs(self.prev_state[1]) - abs(cur_state[1])) \
                         - self.dis_rew_factor * abs(cur_state[2])
                self.prev_state = cur_state
            except:
                import pdb
                pdb.set_trace()

        return reward

    def reset(self):
        state, images = self.env.reset(0)
        print('reset count: {}, timestep this episode: {}'.format(self.reset_count, self.timestep))

        timestamp = time.time()
        if RENDER:
            self.video = cv2.VideoWriter(self.VIDEO_LOG_DIR + "{}_sample{}.avi".format(timestamp, self.reset_count),
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0, (1500, 720), True)
        self.timestep = 0
        self.reset_count += 1
        state = np.array(state)
        self.prev_state = state  # dim=3

        self.accum_actions = np.zeros(4)
        return np.concatenate((np.array(state), self.accum_actions))
        # return np.array(state)

    def render(self):
        self.env.render()


if __name__ == '__main__':
    env = TendonGymEnv()
    e = np.array([[1]])  # Max control input. Set too low can lead to Cholesky failures.
    T = 10
    maxiter = 10
    T_sim = 300
    buffer_size = 600
    verbose = True

    X, Y, _, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T_sim, render=False, verbose=verbose)
    for i in range(1, 1):
        X_, Y_, _, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T_sim, render=False, verbose=verbose)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    m_init = np.reshape(np.zeros(state_dim), (1, state_dim))  # initial state mean

    # controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=40)
    controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    pilco = PILCO((X, Y), controller=controller, horizon=T, m_init=m_init)
    pilco.controller.max_action = e

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)
        # model.likelihood.fixed=True

    return_lst = []
    task_length = []
    for rollouts in range(100):
        print("**** ITERATION no.", rollouts, " ****")
        try:
            pilco.optimize_models(maxiter=maxiter)
            pilco.optimize_policy(maxiter=maxiter)
        except:
            # for i in range(len(return_lst)):
            #     return_lst[i] = str(return_lst[i])
            # df = pd.DataFrame(return_lst, columns=['Return per epoch'])
            # df.to_csv(('/home/lab/Github/PILCO/log/Return/{}.csv'.format(time.time())))
            print('Start Error!!!!!!!!!!!!!!!!!!')

        X_new, Y_new, _, ep_return_lst, ep_length = rollout(env=env, pilco=pilco, timesteps=1000, render=False, verbose=verbose)
        return_lst.append(ep_return_lst)
        task_length.append(ep_length)
        # Update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))

        print(len(X))
        if len(X) > buffer_size:
            XY = np.hstack((X, Y))
            row_rand_array = np.arange(XY.shape[0])

            np.random.shuffle(row_rand_array)

            XY = XY[row_rand_array[0:buffer_size]]
            print('XY: ', len(XY))
            X = XY[:, :(state_dim + control_dim)]
            Y = XY[:, (state_dim + control_dim):]
            print('X: {}, Y: {}'.format(len(X), len(Y)))
        pilco.mgpr.set_data((X, Y))
    for i in range(len(return_lst)):
        return_lst[i] = str(return_lst[i])
    df = pd.DataFrame({'Return per epoch': return_lst, 'Task length per epoch': task_length})
    # df = pd.DataFrame(([return_lst, task_length]), columns=['Return per epoch', 'Task length per epoch'])
    df.to_csv(('/home/lab/Github/PILCO/log/Return/{}.csv'.format(time.time())))

