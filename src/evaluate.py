import os

import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.environment import make_env
from src.planner import Agent, make_reward_objective, make_vlm_objective

METHOD_NAMES = {
    "random": "random",
    "reward": "world-model planning (no VLM)",
    "vlm": "world-model planning + VLM",
}


def run_random_episode(env, seed):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done, steps, success = False, 0, False
    while not done:
        _, _, term, trunc, _ = env.step(int(rng.choice(config.ACTIONS)))
        steps += 1; done = term or trunc; success = term
    return success, steps


def run_planning_episode(env, wm, objective, seed):
    obs, _ = env.reset(seed=seed)
    agent = Agent(wm, objective); agent.reset(obs)
    done, steps, success = False, 0, False
    while not done:
        a = agent.act()
        obs, _, term, trunc, _ = env.step(a); agent.observe(a, obs)
        steps += 1; done = term or trunc; success = term
    return success, steps


def _episode(env, wm, method, seed, scorer):
    if method == "random":
        return run_random_episode(env, seed)
    if method == "reward":
        return run_planning_episode(env, wm, make_reward_objective(wm), seed)
    if method == "vlm":
        return run_planning_episode(env, wm, make_vlm_objective(wm, scorer), seed)
    raise ValueError(f"unknown method: {method}")


def evaluate_method(wm, method, scorer=None):
    env = make_env(eval_max_steps=config.EVAL_MAX_STEPS)
    per_seed = []
    for sd in config.EVAL_SEEDS:
        results = [_episode(env, wm, method, 1000 * sd + i, scorer) for i in range(config.EVAL_EPISODES)]
        per_seed.append(float(np.mean([s for s, _ in results])))
    env.close()
    return per_seed

def evaluate_all(wm, scorer):
    rows = []
    for method in ["random", "reward", "vlm"]:
        sc = scorer if method == "vlm" else None
        per_seed = evaluate_method(wm, method, sc)
        rows.append((METHOD_NAMES[method], float(np.mean(per_seed)), float(np.std(per_seed))))
        print(f"{METHOD_NAMES[method]:32s} success_rate = {np.mean(per_seed):.2f}")
    return pd.DataFrame(rows, columns=["method", "success_rate", "std"])

def record_gif(wm, method, seed, path, scorer=None):
    env = make_env(eval_max_steps=config.EVAL_MAX_STEPS)
    obs, _ = env.reset(seed=seed)
    objective = make_vlm_objective(wm, scorer) if method == "vlm" else make_reward_objective(wm)
    agent = Agent(wm, objective); agent.reset(obs)
    frames = [np.asarray(env.render())]
    done, success = False, False
    while not done:
        a = agent.act()
        obs, _, term, trunc, _ = env.step(a); agent.observe(a, obs)
        frames.append(np.asarray(env.render()))
        done = term or trunc; success = term
    env.close()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=4, loop=0)
    return success, len(frames) - 1

def plot_success_rates(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(df["method"], df["success_rate"], yerr=df["std"], capsize=5)
    plt.ylabel("success rate"); plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.title("Success rate по методам")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_training_loss(history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    steps = [h["step"] for h in history]
    recon = [h["recon"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, recon)
    plt.xlabel("шаг обучения"); plt.ylabel("reconstruction loss")
    plt.title("Обучение модели мира")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
