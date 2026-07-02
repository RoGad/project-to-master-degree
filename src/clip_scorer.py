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

    @torch.no_grad()
    def score_frames(self, frames01):
        x = F.interpolate(frames01, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * (feat @ self.text_feat.T)  # (M,2): [цель, негатив]
        return logits.softmax(-1)[:, 0]  # вероятность 'цель'