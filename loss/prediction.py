import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def sim(tau, img_embs, text_embs, cap_lens, mode='sim'):
    text_embs = text_embs.float() 
    img_embs = img_embs.float() 
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = logit_scale.exp()
    text_embs = text_embs/ text_embs.norm(dim=-1, keepdim=True)
    img_embs = img_embs/ img_embs.norm(dim=-1, keepdim=True)
    logits_per_image = logit_scale * img_embs @ text_embs.t()
    logits_per_text = logits_per_image.t()
    raw_sims = 0.5 * (logits_per_text+logits_per_image)
    if mode == 'sim':
        return torch.sigmoid(raw_sims)
    else:
        sims, evidences, sims_tanh = torch.sigmoid(raw_sims), torch.exp(torch.tanh(raw_sims)/ tau), torch.tanh(raw_sims)
        return sims, evidences, sims_tanh  

def get_alpha(tau, img_embs, text_embs, lengths):
    sims, evidences, sims_tanh =  sim(tau,img_embs, text_embs, lengths, 'not sims')
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    return alpha_i2t, alpha_t2i, norm_e, sims_tanh, sims

def prediction(img_embs, text_embs, lengths, tau,device):
    bs = img_embs.size(0)
    alpha_i2t, alpha_t2i, norm_e, sims_tanh, _ = get_alpha(tau,img_embs, text_embs, lengths)
    g_t = torch.from_numpy(np.array([i for i in range(bs)])).to(device)
    pred = g_t.eq(torch.argmax(norm_e, dim=1)) + 0
    pred = pred.to(device)
    n_idx = (1 - pred).nonzero().view(1, -1)[0].tolist()
    c_idx = pred.nonzero().view(1, -1)[0].tolist()
    print(alpha_i2t.shape,alpha_t2i.shape)
    min_n = torch.argmin(norm_e[n_idx], axis=1)
    min_c = torch.argmin(norm_e[:,n_idx], axis=0)
    max_n = torch.argmax(alpha_i2t[n_idx], axis=1)
    max_c = torch.argmax(alpha_t2i[n_idx], axis=1)
    print(n_idx)
    print(min_n,min_c)
    print(max_n,max_c)
    min_n = torch.argmin(alpha_i2t[n_idx], axis=1)
    min_c = torch.argmin(alpha_t2i[n_idx], axis=1)
    print(min_n,min_c)
    return n_idx,c_idx,min_n,min_c