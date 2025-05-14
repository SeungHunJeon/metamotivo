# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence, Callable
import dataclasses
import multiprocessing
import gymnasium
import numpy as np
import functools
import torch
from humenv.misc.motionlib import MotionBuffer
from metamotivo.utils.metrics import distance_proximity, emd, phc_metrics, emd_numpy
from humenv.bench.gym_utils.episodes import Episode
from humenv import make_humenv, CustomManager
from concurrent.futures import ProcessPoolExecutor
from packaging.version import Version
from tqdm import tqdm
import os
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32))
else:

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32), env.observation_space)


@dataclasses.dataclass(kw_only=True)
class TrackingEvaluation:
    motions: str | List[str]
    motion_base_path: str | None = None
    # environment parameters
    num_envs: int = 1
    wrappers: Sequence[Callable[[gymnasium.Env], gymnasium.Wrapper]] = dataclasses.field(
        default_factory=lambda: [gymnasium.wrappers.FlattenObservation, cast_obs_wrapper]
    )
    env_kwargs: dict = dataclasses.field(default_factory=dict)
    mp_context: str = "forkserver"
    env: VecEnv = None

    # def __post_init__(self) -> None:
    #     if self.num_envs > 1:
    #         self.mp_manager = CustomManager()
    #         self.mp_manager.start()
    #         self.motion_buffer = self.mp_manager.MotionBuffer(
    #             files=self.motions, base_path=self.motion_base_path, keys=["qpos", "qvel", "observation"]
    #         )
    #     else:
    #         self.mp_manager = None
    #         self.motion_buffer = MotionBuffer(files=self.motions, base_path=self.motion_base_path, keys=["qpos", "qvel", "observation"])

    def run(self, agent: Any) -> Dict[str, Any]:
        ids = self.env.motion_buffer.get_motion_ids()
        motions_per_worker = np.array_split(ids, len(ids) / self.env.num_envs)

        metrics = {}
        for motions in motions_per_worker:
            metric = self.tracking((motions, agent))
            metrics.update(metric)
        return metrics

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()

    def tracking(self, inputs):
        motion_ids, agent = inputs
        # env = make_humenv(num_envs=1, wrappers=wrappers, **env_kwargs)[0]
        metrics = {}

        eps_ = self.env.motion_buffer.get(motion_ids)

        max_len = max(seq_len for seq_len in eps_["seq_len"])

        ctxs = torch.zeros([max_len - 1, self.env.num_envs, agent.cfg.archi.z_dim])
        truncates = np.zeros([max_len - 1, self.env.num_envs, 1], dtype=bool)
        gc = np.zeros([self.env.num_envs, self.env.num_acts + 7], dtype=np.float32)
        gv = np.zeros([self.env.num_envs, self.env.num_acts + 6], dtype=np.float32)
        tracking_targets = np.zeros([max_len - 1, self.env.num_envs, agent.cfg.obs_dim], dtype=np.float32)
        for id, (observations, seq_len, qpos, qvel) in enumerate(zip(eps_["observation"], eps_["seq_len"], eps_["qpos"], eps_["qvel"])):
            tracking_target = observations[1:]
            ctx = agent.tracking_inference(next_obs=tracking_target)
            ctx = [None] * tracking_target.shape[0] if ctx is None else ctx
            ctxs[:seq_len - 1, id, :] = ctx
            gc[id] = qpos[0]
            gv[id] = qvel[0]

            tracking_targets[:seq_len - 1, id, :] = tracking_target
            truncates[seq_len - 1:, id, 0] = True  # seq_len 이후 부분을 True로 설정

        self.env.set_state(gc, gv)
        self.env.integrate()
        td, info = self.env.observe(False)
        observation = td["obs"]
        _episode = Episode()
        _episode.initialise(observation, info)

        for i in range(max_len - 1):
            action = agent.act(observation, ctxs[i])
            reward, terminated, truncated = self.env.step(action)
            td, info = self.env.observe(False)
            observation = td["obs"]
            # observation, reward, terminated, truncated, info = env.step(action)
            _episode.add(observation, reward, action, terminated, truncates[i], info)
        tmp = _episode.get()
        tmp["tracking_target"] = tracking_targets
        tmp["motion_id"] = motion_ids
        tmp["motion_file"] = self.env.motion_buffer.get_name(motion_ids)
        metrics.update(_calc_metrics(tmp))
        return metrics


def _calc_metrics(ep):
    metr = {}
    next_obs = torch.tensor(ep["observation"][1:], dtype=torch.float32)
    truncated = torch.tensor(ep["truncated"], dtype=torch.bool)
    tracking_target = torch.tensor(ep["tracking_target"], dtype=torch.float32)
    dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target, mask=truncated)
    emd_res = emd_numpy(next_obs=next_obs, tracking_target=tracking_target, mask=truncated)
    phc_res = phc_metrics(next_obs=next_obs, tracking_target=tracking_target, mask=truncated)

    for i in range(len(emd_res)):
        motion_file = ep["motion_file"][i]
        metr[motion_file] = {}

        metr[motion_file]["motion_id"] = ep["motion_id"][i]

        # Store the results for each metric
        metr[motion_file]["proximity"] = dist_prox_res["proximity"][i]  # Assuming proximity is a list
        metr[motion_file]["distance"] = dist_prox_res["distance"][i]    # Assuming distance is a list
        metr[motion_file]["success"] = dist_prox_res["success"][i]      # Assuming success is a list

        metr[motion_file]["emd"] = emd_res["emd"][i]  # Assuming emd is a list
        metr[motion_file]["mpjpe_g"] = phc_res["mpjpe_g"][i]  # Assuming mpjpe_g is a list
        metr[motion_file]["vel_dist"] = phc_res["vel_dist"][i]  # Assuming vel_dist is a list
        metr[motion_file]["accel_dist"] = phc_res["accel_dist"][i]  # Assuming accel_dist is a list
        metr[motion_file]["success_phc_linf"] = phc_res["success_phc_linf"][i]  # Assuming success_phc_linf is a list
        metr[motion_file]["success_phc_mean"] = phc_res["success_phc_mean"][i]  # Assuming success_phc_mean is a list
    return metr
