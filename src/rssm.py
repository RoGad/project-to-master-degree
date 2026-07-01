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