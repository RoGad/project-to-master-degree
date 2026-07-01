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

def sample_batch(buffer):
    valid = [ep for ep in buffer.episodes if len(ep[1]) >= config.SEQ_LEN]
    frames_list, actions_list, rewards_list = [], [], []
    for _ in range(config.BATCH_SIZE):
        obs, act, rew = valid[np.random.randint(len(valid))]
        i = np.random.randint(0, len(act) - config.SEQ_LEN + 1)
        frames_list.append(obs[i:i + config.SEQ_LEN])
        pa = torch.zeros(config.SEQ_LEN, config.NUM_ACTIONS)
        pa[1:] = F.one_hot(torch.as_tensor(act[i:i + config.SEQ_LEN - 1]), config.NUM_ACTIONS).float()
        actions_list.append(pa)
        rewards_list.append(torch.as_tensor(rew[i:i + config.SEQ_LEN]))
    arr = np.stack(frames_list)
    frames = preprocess(arr.reshape(config.BATCH_SIZE * config.SEQ_LEN, *arr.shape[2:]))
    frames = frames.reshape(config.BATCH_SIZE, config.SEQ_LEN, 3, config.IMG_SIZE, config.IMG_SIZE)
    return frames, torch.stack(actions_list).to(config.DEVICE), torch.stack(rewards_list).to(config.DEVICE)


def train_world_model(wm, buffer, steps=None, log_every=200):
    steps = steps or config.TRAIN_STEPS
    opt = torch.optim.Adam(wm.parameters(), config.LEARNING_RATE)
    for it in range(steps):
        frames, prev_actions, rewards = sample_batch(buffer)
        total, info = compute_loss(wm, frames, prev_actions, rewards)
        opt.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), config.GRAD_CLIP)
        opt.step()
        if it % log_every == 0 or it == steps - 1:
            print(f"шаг {it:4d} | recon {info['recon']:.1f} | kl {info['kl']:.2f} | reward {info['reward']:.4f}")
    return wm
