import torch
import torch.nn.functional as F

from src import config
from src.environment import preprocess
from src.world_model import imagine_rollout


@torch.no_grad()
def plan(wm, deter, stoch, objective):
    N, H = config.NUM_CANDIDATES, config.HORIZON
    seqs = torch.randint(0, config.NUM_ACTIONS, (N, H), device=config.DEVICE)
    states = imagine_rollout(wm, deter.repeat(N, 1), stoch.repeat(N, 1), seqs)
    scores = objective(states)
    return config.ACTIONS[seqs[scores.argmax(), 0].item()]


def make_reward_objective(wm):
    @torch.no_grad()
    def objective(states):
        return sum(wm.predict_reward(d, s) for d, s in states)
    return objective

def make_vlm_objective(wm, scorer):
    @torch.no_grad()
    def objective(states):
        return scorer.score_rollout(wm, states)
    return objective

class Agent:
    def __init__(self, wm, objective):
        self.wm = wm
        self.objective = objective
        self.deter = None
        self.stoch = None

    @torch.no_grad()
    def reset(self, obs):
        self.deter, self.stoch = self.wm.initial_state(1)
        zero_action = torch.zeros(1, config.NUM_ACTIONS, device=config.DEVICE)
        embed = self.wm.encode(preprocess(obs))
        self.deter, self.stoch, _, _ = self.wm.obs_step(self.stoch, zero_action, self.deter, embed)

    @torch.no_grad()
    def act(self):
        return plan(self.wm, self.deter, self.stoch, self.objective)

    @torch.no_grad()
    def observe(self, action, obs):
        pa = F.one_hot(torch.tensor([config.ACTIONS.index(action)], device=config.DEVICE),
                       config.NUM_ACTIONS).float()
        embed = self.wm.encode(preprocess(obs))
        self.deter, self.stoch, _, _ = self.wm.obs_step(self.stoch, pa, self.deter, embed)
