import torch

def gram_matrix(feat):
    B, C, H, W = feat.size()
    feat = feat.view(B, C, H * W)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    gram /= (C * H * W)
    return gram

def precompute_style_grams(style_img, vgg, layer_cfg):
    with torch.no_grad():
        style_feats = vgg(style_img)
        style_grams = {
            layer: gram_matrix(style_feats[layer])
            for layer in layer_cfg["style"]
        }
        return style_grams
