# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os

import h5py
from typing import Dict, List
import collections
import tqdm
import numbers
from pathlib import Path
from humenv.misc.motionlib import MotionBuffer

class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, fall_prob=0.0):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.cfg = cfg
        self.normalize_ob = normalize_ob
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._gc = np.zeros([self.num_envs, self.num_acts + 7], dtype=np.float32)
        self._gv = np.zeros([self.num_envs, self.num_acts + 6], dtype=np.float32)
        self._gc_init = np.zeros([self.num_envs, self.num_acts + 7], dtype=np.float32)
        self._gv_init = np.zeros([self.num_envs, self.num_acts + 6], dtype=np.float32)
        self._step_count = np.zeros(self.num_envs, dtype=np.int32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._evaluated_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        self.fall_prob = fall_prob

        # Motion buffer
        self.motion_buffer = MotionBuffer(files=self.cfg.motions,
                                          base_path=self.cfg.motions_root,
                                          keys=["qpos", "qvel", "observation"])
    def integrate(self):
        self.wrapper.integrate()

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_task(self, task=None):
        self.wrapper.setTask(task)

    def set_state(self, gc, gv):
        self.wrapper.setState(gc, gv)

    def evaluate_reward(self, q_pos, q_vel, action):
        self.wrapper.evaluateReward(q_pos, q_vel, action, self._evaluated_reward)
        return self._evaluated_reward

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy(), self._truncated.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        self.wrapper.observeGc(self._gc)
        self.wrapper.observeGv(self._gv)
        self.wrapper.observeStepCounter(self._step_count)
        return {"obs":self._observation, "time": np.expand_dims(self._step_count, axis=-1)}, {"qpos": self._gc, "qvel": self._gv}

    def observe_gc(self):
        self.wrapper.observeGc(self._gc)
        return self._gc

    def observe_gv(self):
        self.wrapper.observeGv(self._gv)
        return self._gv

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)

        # Here, we sample the gc init & gv init from motion buffer
        batch = self.motion_buffer.sample(self.num_envs)
        self._gc_init, self._gv_init = batch["qpos"], batch["qvel"]

        self.wrapper.reset(self._gc_init, self._gv_init)

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
