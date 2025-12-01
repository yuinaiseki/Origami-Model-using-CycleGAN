################################################################################
# 1. IMPORTS
################################################################################
import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms 
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

################################################################################
# 2. PASTE YOUR MODEL DEFINITIONS HERE
################################################################################
LAYER_INDICES = {
    'conv1_1': '0', 
    'conv1_2': '2', 
    'conv2_1': '5', 
    'conv2_2': '7',
    'conv3_1': '10', 
    'conv3_2': '12', 
    'conv3_3': '14', 
    'conv3_4': '16',
    'conv4_1': '19', 
    'conv4_2': '21', 
    'conv4_3': '23', 
    'conv4_4': '25',
    'conv5_1': '28', 
    'conv5_2': '30', 
    'conv5_3': '32', 
    'conv5_4': '34'
}

LAYER_CONFIGS = {
    'gatys': {
        'content': ['conv4_2'],
        'style': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        'style_weights': {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        },
    }
}
ACTIVE_LAYER_CONFIG = 'gatys'

class VGGFeatures(nn.Module):
    def __init__(self, vgg, layer_indices):
        super().__init__()
        self.vgg = vgg
        self.idx_to_name = {int(idx_str): name for name, idx_str in layer_indices.items()}
    def forward(self, x):
        cur = x
        feats = {}
        for idx, layer in self.vgg._modules.items():
            cur = layer(cur)
            i = int(idx)
            if i in self.idx_to_name:
                feats[self.idx_to_name[i]] = cur
        return feats

# Load VGG
vgg_model = models.vgg19(pretrained=True).features.to(device).eval()
for p in vgg_model.parameters():
    p.requires_grad = False

vgg = VGGFeatures(vgg_model, LAYER_INDICES).to(device).eval()

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True)
    def forward(self, x):
        return F.relu(self.inorm(self.conv(x)))
       
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Identity(),                            # block.0  <-- no checkpoint entry
            nn.Conv2d(channels, channels, 3, 1, 1),   # block.1  <-- matches
            nn.InstanceNorm2d(channels, affine=True), # block.2  <-- matches
            nn.Identity(),                            # block.3  <-- no checkpoint entry
            nn.Identity(),                            # block.4  <-- no checkpoint entry
            nn.Conv2d(channels, channels, 3, 1, 1),   # block.5  <-- matches
            nn.InstanceNorm2d(channels, affine=True), # block.6  <-- matches
            nn.ReLU(inplace=True)                     # block.7 but no checkpoint entry
        )
    def forward(self, x):
        return x + self.block(x)
    
class UpsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel, upsample=None):
        super().__init__()
        padding = kernel // 2

        # EXACT MATCH FOR CHECKPOINT
        self.deconv = nn.ConvTranspose2d(
            in_c, out_c,
            kernel_size=kernel,
            stride=upsample,
            padding=padding,
            output_padding=(upsample - 1) if upsample else 0
        )

        self.inorm = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        x = self.deconv(x)
        return F.relu(self.inorm(x))


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = ConvLayer(3,   32, 9, 1)
        self.c2 = ConvLayer(32,  64, 3, 2)
        self.c3 = ConvLayer(64, 128, 3, 2)

        self.r1 = ResidualBlock(128)
        self.r2 = ResidualBlock(128)
        self.r3 = ResidualBlock(128)
        self.r4 = ResidualBlock(128)
        self.r5 = ResidualBlock(128)

        self.u1 = UpsampleConv(128, 64, 3, upsample=2)
        self.u2 = UpsampleConv(64, 32, 3, upsample=2)

        self.conv_out = nn.Conv2d(32, 3, 9, 1, 4)

    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)

        y = self.r1(y)
        y = self.r2(y)
        y = self.r3(y)
        y = self.r4(y)
        y = self.r5(y)

        y = self.u1(y)
        y = self.u2(y)

        y = self.conv_out(y)
        return torch.sigmoid(y)



###############################################
# SETTINGS
###############################################

CHECKPOINT_DIR = "../checkpoints_nststyle"   # your folder
CONTENT_IMAGE  = "../test_imgs/butterfly.jpg"     # pick any one content img
STYLE_IMAGE    = "../test_imgs/butterfly.jpg"       # pick the same style img used in training
SAVE_DIR       = "evaluate/normal"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 5. UTILS
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    F = tensor.view(c, h*w)
    return torch.mm(F, F.t())

def to_numpy(x):
    return np.transpose(np.clip(x.squeeze(0).detach().cpu().numpy(), 0, 1), (1,2,0))

# ------------------------------------------------------------
# 6. LOAD CONTENT + STYLE
# ------------------------------------------------------------
content = transform(Image.open(CONTENT_IMAGE)).unsqueeze(0).to(device)
style   = transform(Image.open(STYLE_IMAGE)).unsqueeze(0).to(device)

# ------------------------------------------------------------
# 7. PRECOMPUTE STYLE FEATURES
# ------------------------------------------------------------
cfg = LAYER_CONFIGS[ACTIVE_LAYER_CONFIG]

with torch.no_grad():
    style_feats = vgg(style)

style_grams = {
    layer: gram_matrix(style_feats[layer])
    for layer in cfg["style"]
}

# ------------------------------------------------------------
# 8. STORAGE
# ------------------------------------------------------------
metrics = {
    "epoch": [],
    "ssim": [],
    "psnr": [],
    "gram_distance": [],
    "total_loss": []
}

# ------------------------------------------------------------
# 9. MAIN EVALUATION LOOP
# ------------------------------------------------------------
ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])

for ck in ckpts:
    

    match = re.search(r'epoch(\d+)', ck)
    if match:
        epoch = int(match.group(1))
    else:
        raise ValueError(f"Cannot extract epoch number from filename: {ck}")

    print("Evaluating epoch", epoch)


    model = TransformerNet().to(device)
    state = torch.load(os.path.join(CHECKPOINT_DIR, ck), map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()

    with torch.no_grad():
        out = model(content)
        feats = vgg(out)

    # SSIM
    ssim_val = ssim(to_numpy(content), to_numpy(out), channel_axis=2, data_range=1.0)

    # PSNR
    psnr_val = psnr(to_numpy(content), to_numpy(out))

    # Gram Distance
    gram_dist = 0
    for layer in cfg["style"]:
        gram_out = gram_matrix(feats[layer])
        gram_dist += torch.norm(gram_out - style_grams[layer]).item()

    # Perceptual Loss
    # Content loss
    Lc = torch.nn.functional.mse_loss(
        feats[cfg["content"][0]],
        vgg(content)[cfg["content"][0]]
    )
    # Style loss
    Ls = 0
    for layer in cfg["style"]:
        Ls += cfg["style_weights"][layer] * torch.nn.functional.mse_loss(
            gram_matrix(feats[layer]),
            style_grams[layer]
        )
    # TV loss
    Ltv = torch.mean((out[:, :, :-1, :] - out[:, :, 1:, :])**2) + \
          torch.mean((out[:, :, :, :-1] - out[:, :, :, 1:])**2)

    total = (Lc + 1e6 * Ls + 1e-6 * Ltv).item()

    metrics["epoch"].append(epoch)
    metrics["ssim"].append(ssim_val)
    metrics["psnr"].append(psnr_val)
    metrics["gram_distance"].append(gram_dist)
    metrics["total_loss"].append(total)

# ------------------------------------------------------------
# 10. SAVE JSON
# ------------------------------------------------------------
metrics_clean = {
    k: [float(v) for v in metrics[k]] if isinstance(metrics[k], list) else metrics[k]
    for k in metrics
}

with open(os.path.join(SAVE_DIR, "metrics_baseline.json"), "w") as f:
    json.dump(metrics_clean, f, indent=4)

# ------------------------------------------------------------
# 11. PLOT (LATEX FRIENDLY)
# ------------------------------------------------------------
def plot(metric_name, values, epochs):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name + " vs Epoch")
    plt.grid(True)
    plt.tight_layout()

    fname = metric_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(SAVE_DIR, fname + ".png"), dpi=300)
    plt.savefig(os.path.join(SAVE_DIR, fname + ".pdf"))  # For LaTeX
    plt.close()

plot("SSIM", metrics["ssim"], metrics["epoch"])
plot("PSNR", metrics["psnr"], metrics["epoch"])
plot("Gram Distance", metrics["gram_distance"], metrics["epoch"])
plot("Total Loss", metrics["total_loss"], metrics["epoch"])

print("All plots saved in:", SAVE_DIR)