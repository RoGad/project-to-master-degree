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