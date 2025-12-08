import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import re

# ------------------------------------------------------------
# 1. DEVICE
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# 2. LOAD CONTENT IMAGE
# ------------------------------------------------------------
IMG_PATH = "../test_imgs/butterfly.jpg"   # change to your test image

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

content_img = transform(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0).to(device)

# ------------------------------------------------------------
# 3. VGG FEATURE MODULES
# ------------------------------------------------------------
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
    'conv5_4': '34',
}

LAYER_CONFIG = {
    "content": ['conv4_2'],
    "style": ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    "style_weights": {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1,
    }
}

class VGGFeatures(nn.Module):
    def __init__(self, vgg, layer_indices):
        super().__init__()
        self.vgg = vgg
        self.idx_to_name = {int(idx): name for name, idx in layer_indices.items()}

    def forward(self, x):
        feats = {}
        cur = x
        for idx, layer in self.vgg._modules.items():
            cur = layer(cur)
            idx_int = int(idx)
            if idx_int in self.idx_to_name:
                feats[self.idx_to_name[idx_int]] = cur
        return feats

vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
vgg = VGGFeatures(vgg_model, LAYER_INDICES).to(device)
for p in vgg.parameters(): p.requires_grad = False

# ------------------------------------------------------------
# 4. GRAM MATRIX
# ------------------------------------------------------------
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (ch * h * w)

# ------------------------------------------------------------
# 5. TRANSFORMERNET (checkpoint-compatible)
# ------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True)
    def forward(self, x):
        return F.relu(self.inorm(self.conv(x)))

class StylizedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1   = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2   = nn.InstanceNorm2d(channels, affine=True)
        self.style_gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
    def forward(self, x):
        out  = F.relu(self.in1(self.conv1(x)))
        out  = self.in2(self.conv2(out))
        gate = self.style_gate(out)
        return out * gate + x

class UpsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel, upsample=None):
        super().__init__()
        self.upsample = upsample
        padding = kernel // 2
        self.conv  = nn.Conv2d(in_c, out_c, kernel, 1, padding)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True)
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return F.relu(self.inorm(self.conv(x)))

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.res1 = StylizedResidualBlock(128)
        self.res2 = StylizedResidualBlock(128)
        self.res3 = StylizedResidualBlock(128)
        self.res4 = StylizedResidualBlock(128)
        self.res5 = StylizedResidualBlock(128)
        self.up1 = UpsampleConv(128, 64, 3, upsample=2)
        self.up2 = UpsampleConv(64, 32, 3, upsample=2)
        self.conv_out = nn.Conv2d(32, 3, 9, 1, 4)
    def forward(self, x):
        y = self.conv1(x); y = self.conv2(y); y = self.conv3(y)
        y = self.res1(y);  y = self.res2(y);  y = self.res3(y); y = self.res4(y); y = self.res5(y)
        y = self.up1(y);   y = self.up2(y)
        y = self.conv_out(y)
        return torch.sigmoid(y)

# ------------------------------------------------------------
# 6. EVALUATION OVER EPOCH CHECKPOINTS
# ------------------------------------------------------------

CKPT_DIR = "../checkpoints_nststyle"
def extract_epoch(filename):
    m = re.search(r'epoch(\d+)', filename)
    return int(m.group(1)) if m else -1

ckpts = sorted(
    [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")],
    key=extract_epoch
)

metrics = {
    "epochs": [],
    "loss": [],
    "ssim": [],
    "psnr": [],
    "gram": []
}

def to_numpy(t):
    return t.squeeze(0).permute(1,2,0).detach().cpu().numpy()

# ------------------------------------------------------------
# 7. COMPUTE METRICS
# ------------------------------------------------------------
for ck in ckpts:
    if not ck.endswith(".pth"):
        continue

  
    match = re.search(r'epoch(\d+)', ck)
    if not match:
        continue

    epoch = int(match.group(1))
    print("Evaluating epoch", epoch)

    model = TransformerNet().to(device).eval()
    state = torch.load(os.path.join(CKPT_DIR, ck), map_location=device)
    model.load_state_dict(state)

    # Stylize the content image
    with torch.no_grad():
        out = model(content_img)

    # Total loss (content + style)
    content_feats = vgg(content_img)
    out_feats = vgg(out)

    # Content loss
    content_loss = F.mse_loss(out_feats["conv4_2"], content_feats["conv4_2"])

    # Style loss
    style_loss = 0.0
    for layer in LAYER_CONFIG["style"]:
        G_o = gram_matrix(out_feats[layer])
        style_loss += LAYER_CONFIG["style_weights"][layer] * F.mse_loss(G_o, G_o.detach())

    total_loss = float(content_loss + style_loss)

    # SSIM & PSNR
    ssim_val = ssim(to_numpy(content_img), to_numpy(out), channel_axis=2, data_range=1.0)
    psnr_val = psnr(to_numpy(content_img), to_numpy(out), data_range=1.0)

    # Gram distance
    Gc = gram_matrix(content_feats["conv4_2"])
    Go = gram_matrix(out_feats["conv4_2"])
    gram_val = float(F.mse_loss(Gc, Go).item())

    # Record
    metrics["epochs"].append(epoch)
    metrics["loss"].append(total_loss)
    metrics["ssim"].append(float(ssim_val))
    metrics["psnr"].append(float(psnr_val))
    metrics["gram"].append(float(gram_val))

# ------------------------------------------------------------
# 8. SAVE METRICS
# ------------------------------------------------------------
os.makedirs("evaluate/stylized", exist_ok=True)

with open("evaluate/stylized/metrics_stylized.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 9. GENERATE LATEX-FRIENDLY PLOTS
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(metrics["epochs"], metrics["loss"], marker="o")
plt.title("Total Loss vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("Total Loss")
plt.grid(True)
plt.savefig("evaluate/stylized/loss_vs_epoch.png", dpi=300)

plt.figure(figsize=(8,5))
plt.plot(metrics["epochs"], metrics["ssim"], marker="o")
plt.title("SSIM vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("SSIM")
plt.grid(True)
plt.savefig("evaluate/stylized/ssim_vs_epoch.png", dpi=300)

plt.figure(figsize=(8,5))
plt.plot(metrics["epochs"], metrics["psnr"], marker="o")
plt.title("PSNR vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("PSNR")
plt.grid(True)
plt.savefig("evaluate/stylized/psnr_vs_epoch.png", dpi=300)

plt.figure(figsize=(8,5))
plt.plot(metrics["epochs"], metrics["gram"], marker="o")
plt.title("Gram Distance vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("Gram Distance")
plt.grid(True)
plt.savefig("evaluate/stylized/gram_vs_epoch.png", dpi=300)

print("Stylized model metrics saved.")
