import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from src import config
from src.environment import preprocess


def compute_loss(wm, frames, prev_actions, rewards):
    B, L = frames.shape[:2]
    embeds = wm.encode(frames.reshape(B * L, 3, config.IMG_SIZE, config.IMG_SIZE)).reshape(B, L, -1)
    deter, stoch = wm.initial_state(B)
    deters, stochs, kls = [], [], []
    for t in range(L):
        deter, stoch, prior, post = wm.obs_step(stoch, prev_actions[:, t], deter, embeds[:, t])
        deters.append(deter); stochs.append(stoch)
        kls.append(D.kl_divergence(D.Normal(*post), D.Normal(*prior)).sum(-1))
    deters = torch.stack(deters, 1)
    stochs = torch.stack(stochs, 1)
    df, sf = deters.reshape(B * L, -1), stochs.reshape(B * L, -1)
    recon = wm.decode(df, sf).reshape(B, L, 3, config.IMG_SIZE, config.IMG_SIZE)
    recon_loss = 0.5 * ((recon - frames) ** 2).flatten(2).sum(-1).mean()
    kl_loss = torch.stack(kls, 1).clamp(min=config.FREE_NATS).mean()
    reward_loss = ((wm.predict_reward(df, sf).reshape(B, L) - rewards) ** 2).mean()
    total = recon_loss + kl_loss + reward_loss
    info = {"recon": recon_loss.detach().item(), "kl": kl_loss.detach().item(), "reward": reward_loss.detach().item()}
    return total, info