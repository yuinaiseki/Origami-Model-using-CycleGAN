import torch
import torch.nn.functional as F
from .gram import gram_matrix

def psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim_val = (
        (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    )
    return ssim_val

def gram_distance(img_feats, ref_feats, layers):
    total = 0.0
    for layer in layers:
        g1 = gram_matrix(img_feats[layer])
        g2 = gram_matrix(ref_feats[layer])
        total += F.mse_loss(g1, g2)
    return total
