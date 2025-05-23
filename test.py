from packaging.version import Version
from metamotivo.fb_cpr.model import FBcprModel
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

import torch
import mediapy as media
import math
import h5py
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    model = FBcprModel.load("/home/oem/workspace/metamotivo/examples/tmp_fbcpr/BKC01ND8M5/checkpoint/model")
