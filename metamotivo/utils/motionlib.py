# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import numpy as np
from typing import Dict, List
import collections
import tqdm
import numbers
from pathlib import Path


def load_episode_based_h5(file: str, keys: List[str] | None = None, use_tqdm: bool = False):
    hf = h5py.File(file, "r")
    data = []
    num_ep = hf.attrs["num_episodes"]
    tqdm_bar = tqdm.tqdm(total=num_ep, leave=False) if use_tqdm else None
    for i in range(num_ep):
        episode = hf[f"ep_{i}"]
        keys_ = keys if keys is not None else episode.keys()
        ep = {k: episode[k][:] for k in keys_}
        ep["file_name"] = file
        data.append(ep)
        if tqdm_bar is not None:
            tqdm_bar.update()
    return data


def canonicalize(path: str, base_path: str | None) -> str:
    path = Path(path)
    if not path.is_absolute() and base_path is not None:
        path = Path(base_path) / path
    if not path.exists():
        raise ValueError(f"{path.absolute()} does not exists")
    return str(path)


class MotionBuffer:
    def __init__(
            self,
            files: List[str] | str,
            base_path: str | None = None,
            keys: list[str] = ["qpos", "qvel"],
    ) -> None:
        self.storage, motion_ids, self.file_names, self.seq_len = [], [], [], []
        files = [files] if isinstance(files, str) else files
        if len(files) == 0:
            raise ValueError("Something went wrong. MotionBuffer received no files to load.")
        for f in files:
            if f.endswith("txt"):
                with open(f, "r") as txtf:
                    h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
                episodes = []
                for h5 in tqdm.tqdm(h5files, leave=False):
                    h5 = canonicalize(h5, base_path=base_path)
                    episodes.extend(load_episode_based_h5(h5, keys=None))
            else:
                h5 = canonicalize(f, base_path=base_path)
                episodes = load_episode_based_h5(h5, keys=None, use_tqdm=True)
            for ep in episodes:
                _mid = ep["motion_id"][0].item()
                _e = {k: ep[k] for k in keys}
                _e["motion_id"] = _mid
                self.storage.append(_e)
                motion_ids.append(_mid)
                self.file_names.append(str(Path(ep["file_name"]).name))
                self.seq_len.append(len(ep["qpos"]))

                # Sorting by seq_len
        sorted_indices = np.argsort(self.seq_len)  # seq_len을 기준으로 인덱스 정렬
        self.storage = [self.storage[i] for i in sorted_indices]  # storage 정렬
        self.motion_ids = np.array([motion_ids[i] for i in sorted_indices])  # motion_ids 정렬
        self.file_names = [self.file_names[i] for i in sorted_indices]  # file_names 정렬
        self.seq_len = [self.seq_len[i] for i in sorted_indices]  # seq_len 정렬

        self.priorities = np.ones_like(self.motion_ids) / len(self.motion_ids)

    def sample(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        self.ep_ind = np.random.choice(len(self), p=self.priorities, size=batch_size, replace=True)
        output = collections.defaultdict(list)
        for ep_ in self.ep_ind:
            t_idx = np.random.randint(0, self.storage[ep_]["qpos"].shape[0], 1).item()
            for k, v in self.storage[ep_].items():
                if isinstance(v, numbers.Number):
                    output[k].append(np.array([v]).reshape(1, -1))
                else:
                    output[k].append(v[t_idx].reshape(1, -1))
        return {k: np.concatenate(v, axis=0) for k, v in output.items()}

    def update_priorities(self, motions_id: List | np.ndarray, priorities: List | np.ndarray) -> None:
        for m, p in zip(motions_id, priorities):
            y = self._get_index_of_motion(m)
            self.priorities[y] = p
        self.priorities = self.priorities / np.sum(self.priorities)
        assert all(np.isfinite(self.priorities)), "Priorities should be finite"

    ########################
    # Misc
    ########################
    def __len__(self) -> int:
        return len(self.storage)

    def __repr__(self):
        output = f"MotionBuffer(size={len(self)})"
        return output

    ########################
    # Getters
    ########################
    def get_priorities(self):
        return self.priorities.copy()

    def get_motion_ids(self):
        return self.motion_ids.copy()

    def get(self, motion_id: np.ndarray) -> Dict:
        # motion_id가 하나일 때는 기존 방식대로 처리
        if isinstance(motion_id, np.int32) or isinstance(motion_id, int) or isinstance(motion_id, np.int64):
            y = self._get_index_of_motion(motion_id)
            return self.storage[y]

        # motion_id가 배열일 경우 (여러 motion_id 처리)
        elif isinstance(motion_id, np.ndarray):
            result = {key: [] for key in self.storage[0].keys()}  # 첫 번째 아이템의 keys로 초기화
            result["seq_len"] = []

            # motion_id 배열을 한 번 순회하면서 결과를 채웁니다.
            for m_id in motion_id:
                y = self._get_index_of_motion(m_id)
                for key, value in self.storage[y].items():
                    result[key].append(value)
                result["seq_len"].append(self.seq_len[y])

            # 각 항목들을 리스트로 결합하여 반환
            return result

        # 예외 처리: motion_id가 배열이나 정수가 아닐 때
        else:
            raise ValueError(f"Expected motion_id to be int or numpy.ndarray, got {type(motion_id)}")

    def get_name(self, motion_id: np.ndarray) -> List[str]:
        # motion_id가 하나일 때는 기존 방식대로 처리
        if isinstance(motion_id, (np.int32, np.int64, int)):
            y = self._get_index_of_motion(motion_id)
            return self.file_names[y]

        # motion_id가 배열일 경우 (여러 motion_id 처리)
        elif isinstance(motion_id, np.ndarray):
            # motion_id에 대한 인덱스를 찾아 file_names를 반환
            return [self.file_names[self._get_index_of_motion(m)] for m in motion_id]

        else:
            raise ValueError(f"Expected motion_id to be int or numpy.ndarray, got {type(motion_id)}")

    def get_id(self, motion_name: int) -> str:
        y = np.where(self.file_names == motion_name)[0]
        if len(y) != 1:
            raise ValueError(f"Motion name={motion_name} not found in the buffer")
        return self.motion_ids[y.item()]

    def _get_index_of_motion(self, motion_id: int) -> int:
        y = np.where(self.motion_ids == motion_id)[0]
        if len(y) != 1:
            raise ValueError(f"Motion id={motion_id} not found in the buffer")
        return y.item()
