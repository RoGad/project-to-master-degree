import open_clip
import torch
import torch.nn.functional as F

from src import config


class CLIPScorer:
    def __init__(self, goal_prompt=None, neg_prompt=None):
        goal = goal_prompt or config.GOAL_PROMPT
        neg = neg_prompt or config.NEG_PROMPT
        self.model = open_clip.create_model(config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
        self.model = self.model.to(config.DEVICE).eval()
        tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        with torch.no_grad():
            text = tokenizer([goal, neg]).to(config.DEVICE)
            text_feat = self.model.encode_text(text)
            self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        self.mean = torch.tensor([0.4815, 0.4578, 0.4082], device=config.DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758], device=config.DEVICE).view(1, 3, 1, 1)