import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

ENV_ID = "MiniGrid-Empty-Random-6x6-v0"
IMG_SIZE = 64
ACTIONS = [0, 1, 2]
NUM_ACTIONS = len(ACTIONS)

COLLECT_EPISODES = 150

DETER_DIM = 128
STOCH_DIM = 16
HIDDEN_DIM = 128
FREE_NATS = 3.0

BATCH_SIZE = 32
SEQ_LEN = 25
TRAIN_STEPS = 3000
LEARNING_RATE = 1e-3
GRAD_CLIP = 100.0

NUM_CANDIDATES = 150
HORIZON = 10
SCORE_LAST_K = 3

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
GOAL_PROMPT = "the red agent is on the green goal square"
NEG_PROMPT = "the red agent is far away from the green goal"

EVAL_EPISODES = 15
EVAL_SEEDS = [0, 1, 2]
EVAL_MAX_STEPS = 30

CHECKPOINT_DIR = "checkpoints"
GIF_DIR = "results/gifs"
PLOT_DIR = "results/plots"