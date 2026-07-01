import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from src import config

def make_env(eval_max_steps = None):
    env = gym.make(config.ENV_ID, render_mode="rgb_array")
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    if eval_max_steps is not None:
        env.reset()
        env.unwrapped.max_steps = eval_max_steps
    return env

def preprocess(frames):
    t = torch.as_tensor(np.asarray(frames), dtype=torch.float32, device=config.DEVICE)
    if t.ndim == 3:
        t = t[None]
    t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=config.IMG_SIZE, mode="bilinear", align_corners=False)
    return t - 0.5

def to_image(frame_tensor):
    return (frame_tensor + 0.5).clamp(0, 1).premute(1, 2, 0).cpu().numpy()
