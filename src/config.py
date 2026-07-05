import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

ENV_ID = "MiniWorld-OneRoomS6Fast-v0"
IMG_SIZE = 64
ACTIONS = [0, 1, 2]
NUM_ACTIONS = len(ACTIONS)

COLLECT_EPISODES = 300

DETER_DIM = 128
STOCH_DIM = 16
HIDDEN_DIM = 128
FREE_NATS = 3.0

BATCH_SIZE = 32
SEQ_LEN = 15
TRAIN_STEPS = 3000
LEARNING_RATE = 1e-3
GRAD_CLIP = 100.0

NUM_CANDIDATES = 150
HORIZON = 10
SCORE_LAST_K = 3

CLIP_MODEL = "ViT-B-16-quickgelu"
CLIP_PRETRAINED = "openai"
GOAL_PROMPT = "a large red box close up"
NEG_PROMPT = "a small red box far away"

EVAL_EPISODES = 15
EVAL_SEEDS = [0, 1, 2]
EVAL_MAX_STEPS = 15

CHECKPOINT_DIR = "checkpoints"
GIF_DIR = "results/gifs"
PLOT_DIR = "results/plots"