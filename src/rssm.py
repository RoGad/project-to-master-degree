import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,4,2), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,128,4,2), nn.ReLU(),
            nn.Conv2d(128,256,4,2), nn.ReLU(),
            nn.Flatten(),
        )
        self.embed_dim = 256 * 2 * 2

    def forward(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(config.DETER_DIM + config.STOCH_DIM, 1024)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),   nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),    nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
        )

    def forward(self, feat):
        h = self.fc(feat).view(-1, 1024, 1, 1)
        return self.net(h)

class RSSMCore(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.deter_dim = config.DETER_DIM
        self.stoch_dim = config.STOCH_DIM
        h = config.HIDDEN_DIM
        self.in_proj = nn.Linear(config.STOCH_DIM + config.NUM_ACTIONS, h)
        self.gru = nn.GRUCell(h, config.DETER_DIM)
        self.prior_net = nn.Sequential(
            nn.Linear(config.DETER_DIM, h), nn.ELU(), nn.Linear(h, 2 * config.STOCH_DIM))
        self.post_net = nn.Sequential(
            nn.Linear(config.DETER_DIM + embed_dim, h), nn.ELU(), nn.Linear(h, 2 * config.STOCH_DIM))

    def initial_state(self, batch_size):
        deter = torch.zeros(batch_size, self.deter_dim, device=config.DEVICE)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=config.DEVICE)
        return deter, stoch

    def _dist(self, params):
        mean, std = params.chunk(2, -1)
        return mean, F.softplus(std) + 0.1        # std всегда положительный

    def _recurrent(self, prev_stoch, prev_action, prev_deter):
        x = F.elu(self.in_proj(torch.cat([prev_stoch, prev_action], -1)))
        return self.gru(x, prev_deter)

    def obs_step(self, prev_stoch, prev_action, prev_deter, embed):
        deter = self._recurrent(prev_stoch, prev_action, prev_deter)
        prior_mean, prior_std = self._dist(self.prior_net(deter))
        post_mean, post_std = self._dist(self.post_net(torch.cat([deter, embed], -1)))
        stoch = post_mean + post_std * torch.randn_like(post_std)   # сэмпл из posterior
        return deter, stoch, (prior_mean, prior_std), (post_mean, post_std)

    def img_step(self, prev_stoch, prev_action, prev_deter, sample=False):
        deter = self._recurrent(prev_stoch, prev_action, prev_deter)
        prior_mean, prior_std = self._dist(self.prior_net(deter))
        if sample:
            stoch = prior_mean + prior_std * torch.randn_like(prior_std)
        else:
            stoch = prior_mean
        return deter, stoch

class RewardHead(nn.Module):
    def __init__(self):
        super().__init__()
        h = config.HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(config.DETER_DIM + config.STOCH_DIM, h), nn.ELU(), nn.Linear(h, 1))

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.core = RSSMCore(self.encoder.embed_dim)
        self.decoder = Decoder()
        self.reward_head = RewardHead()

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, deter, stoch):
        return self.decoder(torch.cat([deter, stoch], -1))

    def predict_reward(self, deter, stoch):
        return self.reward_head(torch.cat([deter, stoch], -1))

    def initial_state(self, batch_size):
        return self.core.initial_state(batch_size)

    def obs_step(self, prev_stoch, prev_action, prev_deter, embed):
        return self.core.obs_step(prev_stoch, prev_action, prev_deter, embed)

    def img_step(self, prev_stoch, prev_action, prev_deter, sample=False):
        return self.core.img_step(prev_stoch, prev_action, prev_deter, sample)
