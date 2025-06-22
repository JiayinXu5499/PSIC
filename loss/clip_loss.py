import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class CLIP_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, target,labels):
        query, key = normalize(query, key)
        if target == 1:
            query_similarity = torch.nn.functional.cosine_similarity(query, key)
            loss = query_similarity
        else:
            loss = info_nce(query, key,labels)
        return loss


def info_nce(query, key, labels, temperature=0.1):
    labels = torch.tensor(labels, dtype=torch.long, device=query.device)
    logit_scale         =  nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = logit_scale.exp()
    logits    = logit_scale * query @ key.t()
    #loss = F.cross_entropy(logits, labels)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    #print(loss)
    return loss





def normalize(*xs):
    return [None if x is None else x/x.norm(dim = 1, keepdim = True) for x in xs]