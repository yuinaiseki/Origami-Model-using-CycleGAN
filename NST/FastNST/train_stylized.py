import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from PIL import Image
import torchvision.transforms as T

from models.stylized import TransformerNetStylized
from models.vgg_loss import load_vgg

from utils.dataset import MaskedImageDataset
from utils.loss import content_loss, style_loss, total_variation_loss
from utils.gram import gram_matrix
from utils.metrics import psnr, ssim, gram_distance
from utils.exp_utils import save_checkpoint, save_json, plot_curve
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
RUN_ROOT   = "evaluate/v2/stylized/train"


def load_image(path, device, size=256):
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)

def compute_style_grams(style_path, device, vgg, layer_cfg):

    files = [f for f in os.listdir(style_path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        raise ValueError("No style images found in folder:", style_path)

    gram_acc = {layer: None for layer in layer_cfg["style"]}
    count = 0

    for f in files:
        full = os.path.join(style_path, f)
        img = load_image(full, device)
        feats = vgg(img)

        for layer in layer_cfg["style"]:
            g = gram_matrix(feats[layer])
            if gram_acc[layer] is None:
                gram_acc[layer] = g
            else:
                gram_acc[layer] += g

        count += 1

    for layer in gram_acc:
        gram_acc[layer] /= count

    print(f"[✓] Averaged style grams from {count} style images")
    return gram_acc

def evaluate_on_val(model, val_loader, vgg, layer_cfg, style_grams, device):

    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_gram = 0
    total_loss = 0
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

            total = c_loss + s_loss + tv_loss

            total_psnr += psnr(gen, rgb).item()
            total_ssim += ssim(gen, rgb).item()
            total_gram += gram_distance(gen_feats, rgb_feats, layer_cfg["style"]).item()
            total_loss += total.item()

            count += 1
            

    return {
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "gram": total_gram / count,
        "loss": total_loss / count
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

    model = TransformerNetStylized().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    vgg, layer_cfg = load_vgg(device)

    style_grams = compute_style_grams(STYLE_PATH, device, vgg, layer_cfg)

    history = {}
    best_val_gram = float("inf")
    best_ckpt = os.path.join(run_folder, "best_checkpoint.pth")


    EPOCHS = 12 

    for epoch in range(1, EPOCHS + 1):
        print(f"\n\nEpoch {epoch}/{EPOCHS}\n")

        model.train()
        train_content = train_style = train_tv = train_total = 0
        count = 0

        for combined, rgb in train_loader:
            combined = combined.to(device)
            rgb = rgb.to(device)

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
            "style_loss":   train_style / count,
            "tv_loss":      train_tv / count,
            "total_loss":   train_total / count
        }


        val_metrics = evaluate_on_val(model, val_loader, vgg, layer_cfg, style_grams, device)

        history[f"epoch_{epoch}"] = {
            "train": train_metrics_epoch,
            "val": val_metrics
        }

        save_json(os.path.join(metrics_dir, "train_val_metrics.json"), history)

        if val_metrics["gram"] < best_val_gram:
            best_val_gram = val_metrics["gram"]
            save_checkpoint(model, optimizer, epoch, best_val_gram, best_ckpt)
            print(f"[✓] New BEST checkpoint saved! gram={best_val_gram:.6f}")

        print("Train:", train_metrics_epoch)
        print("Val:  ", val_metrics)

    def extract(metric):
        return [history[f"epoch_{e}"]["val"][metric] for e in range(1, EPOCHS+1)]

    plot_curve(extract("psnr"), "Validation PSNR", "PSNR", os.path.join(plots_dir, "val_psnr.png"))
    plot_curve(extract("ssim"), "Validation SSIM", "SSIM", os.path.join(plots_dir, "val_ssim.png"))
    plot_curve(extract("gram"), "Validation Gram Distance", "Gram", os.path.join(plots_dir, "val_gram.png"))

    print("\n[✓] Training COMPLETED.")


if __name__ == "__main__":
    train()
