import os
import torch
import json
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, val_gram, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_gram": val_gram
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["val_gram"]


def find_best_checkpoint(folder):
    ckpts = [f for f in os.listdir(folder) if f.endswith(".pth")]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda f: float(f.split("gram_")[1].replace(".pth", "")))
    return os.path.join(folder, ckpts[0])

def plot_curve(values, title, ylabel, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r") as f:
        return json.load(f)

