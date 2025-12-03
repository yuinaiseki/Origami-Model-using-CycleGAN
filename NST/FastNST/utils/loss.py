import torch
import torch.nn.functional as F
from .gram import gram_matrix

def content_loss(gen_feats, content_feats, layer_cfg, weight=1.0):
    layer = layer_cfg["content"][0]
    return weight * F.mse_loss(gen_feats[layer], content_feats[layer])

def style_loss(gen_feats, style_grams, layer_cfg):
    loss = 0.0
    for layer in layer_cfg["style"]:
        gen_gram = gram_matrix(gen_feats[layer])
        target_gram = style_grams[layer]
        w = layer_cfg["style_weights"][layer]
        if target_gram.shape[0] == 1 and gen_gram.shape[0] > 1:
            target_gram = target_gram.expand_as(gen_gram)

        loss += w * F.mse_loss(gen_gram, target_gram)
    return loss

def total_variation_loss(img, weight=1e-6):
    b, c, h, w = img.size()
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return weight * (tv_h + tv_w)
