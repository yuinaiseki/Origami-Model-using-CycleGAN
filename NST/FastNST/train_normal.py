import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T

from models.normal import TransformerNetBaseline
from models.vgg_loss import load_vgg

from utils.dataset import MaskedImageDataset
from utils.gram import precompute_style_grams, gram_matrix
from utils.loss import content_loss, style_loss, total_variation_loss
from utils.exp_utils import save_checkpoint, save_json, plot_curve
from utils.metrics import psnr, ssim, gram_distance
from datetime import datetime
import sys

def progress_bar(current, total, bar_length=40):
    fraction = current / total
    filled_len = int(bar_length * fraction)

    bar = "*" * filled_len + "-" * (bar_length - filled_len)
    percent = int(fraction * 100)

    sys.stdout.write(f"\r   [{bar}] {percent:3d}% ({current}/{total})")
    sys.stdout.flush()



TRAIN_PATH = "../../Data/dataset/clean/split/segmented/trainB/butterfly"
VAL_PATH   = "../../Data/dataset/clean/split/segmented/valB/butterfly"
STYLE_PATH = "../../Data/dataset/clean/split/origami/train/butterfly"
RUN_ROOT   = "evaluate/v2/baseline/train"

CHECKPOINT_DIR = "checkpoints/baseline"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 6
EPOCHS     = 2
LR         = 1e-4

def load_image(path, device, size=256):
    tf = T.Compose([T.Resize((size,size)), T.ToTensor()])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)

def compute_style_grams(style_path, device, vgg, layer_cfg):
    files = [f for f in os.listdir(style_path) if f.lower().endswith((".jpg",".png"))]
    acc = {l:None for l in layer_cfg["style"]}
    count=0

    for f in files:
        img = load_image(os.path.join(style_path, f), device)
        feats = vgg(img)
        for layer in layer_cfg["style"]:
            g = gram_matrix(feats[layer])
            if acc[layer] is None: acc[layer]=g
            else: acc[layer]+=g
        count+=1

    for layer in acc:
        acc[layer] /= count

    print(f"[✓] Style grams averaged over {count} images.")
    return acc

def evaluate_on_val(model, val_loader, vgg, layer_cfg, device):

    model.eval()

    val_psnr = 0
    val_ssim = 0
    val_gram = 0
    val_loss_total = 0
    count = 0

    with torch.no_grad():
        for combined, rgb in val_loader:
            combined = combined.to(device)
            rgb = rgb.to(device)

            gen = model(combined)

            gen_feats = vgg(gen)
            rgb_feats = vgg(rgb)

            c_loss = content_loss(gen_feats, rgb_feats, layer_cfg)
            s_loss = style_loss(gen_feats, style_grams, layer_cfg)
            tv_loss = total_variation_loss(gen)

            total_loss = c_loss + s_loss + tv_loss

            val_psnr += psnr(gen, rgb).item()
            val_ssim += ssim(gen, rgb).item()
            val_gram += gram_distance(gen_feats, rgb_feats, layer_cfg["style"]).item()
            val_loss_total += total_loss.item()
            count += 1

    return {
        "psnr": val_psnr / count,
        "ssim": val_ssim / count,
        "gram": val_gram / count,
        "loss": val_loss_total / count
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_folder = RUN_ROOT
    metrics_dir = os.path.join(run_folder, "metrics")
    plots_dir = os.path.join(run_folder, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    train_ds = MaskedImageDataset(TRAIN_PATH)
    val_ds   = MaskedImageDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TransformerNetBaseline().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    vgg, layer_cfg = load_vgg(device)
    global style_grams
    style_grams = compute_style_grams(STYLE_PATH, device, vgg, layer_cfg)

    train_metrics = {}
    best_val_gram = float("inf")
    best_ckpt_path = os.path.join(run_folder, "best_checkpoint.pth")

    EPOCHS = 12  

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_content, train_style, train_tv, train_total = 0, 0, 0, 0
        count = 0

        for combined, rgb in train_loader:
            combined, rgb = combined.to(device), rgb.to(device)

            optimizer.zero_grad()
            gen = model(combined)

            gen_feats = vgg(gen)
            rgb_feats = vgg(rgb)

            c_loss = content_loss(gen_feats, rgb_feats, layer_cfg)
            s_loss = style_loss(gen_feats, style_grams, layer_cfg)
            tv_loss = total_variation_loss(gen)

            total_loss = c_loss + s_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            train_content += c_loss.item()
            train_style += s_loss.item()
            train_tv += tv_loss.item()
            train_total += total_loss.item()
            count += 1
            progress_bar(count, len(train_loader))

        train_metrics_epoch = {
            "content_loss": train_content / count,
            "style_loss": train_style / count,
            "tv_loss": train_tv / count,
            "total_loss": train_total / count
        }

        val_metrics = evaluate_on_val(model, val_loader, vgg, layer_cfg, device)

        train_metrics[f"epoch_{epoch}"] = {
            "train": train_metrics_epoch,
            "val": val_metrics
        }

        save_json(os.path.join(metrics_dir, "train_val_metrics.json"), train_metrics)


        if val_metrics["gram"] < best_val_gram:
            best_val_gram = val_metrics["gram"]
            save_checkpoint(model, optimizer, epoch, best_val_gram, best_ckpt_path)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("Train:", train_metrics_epoch)
        print("Val:  ", val_metrics)

 
    def collect(metric):
        return [train_metrics[f"epoch_{e}"]["val"][metric] for e in range(1, EPOCHS+1)]

    plot_curve(collect("psnr"), "Validation PSNR", "PSNR", os.path.join(plots_dir, "val_psnr.png"))
    plot_curve(collect("ssim"), "Validation SSIM", "SSIM", os.path.join(plots_dir, "val_ssim.png"))
    plot_curve(collect("gram"), "Validation Gram Distance", "Gram", os.path.join(plots_dir, "val_gram.png"))

    print("\n[✓] Training finished.")


if __name__ == "__main__":
    train()