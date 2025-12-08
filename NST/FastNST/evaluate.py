import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T

from models.normal import TransformerNetBaseline
from models.stylized import TransformerNetStylized
from models.vgg_loss import load_vgg

from utils.dataset import MaskedImageDataset
from utils.metrics import psnr, ssim, gram_distance
from utils.exp_utils import save_json, plot_curve
from utils.gram import gram_matrix

TEST_PATH  = "../../Data/dataset/clean/split/segmented/testB/butterfly"
STYLE_PATH = "../../Data/dataset/clean/split/origami/test/butterfly"
RUN_ROOT   = "evaluate/v3"    

def load_image(path, device, size=256):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def compute_style_grams(style_path, device, vgg, layer_cfg):
    files = [f for f in os.listdir(style_path) if f.lower().endswith((".jpg", ".png"))]
    acc = {l: None for l in layer_cfg["style"]}
    count = 0

    for f in files:
        img = load_image(os.path.join(style_path, f), device)
        feats = vgg(img)

        for layer in layer_cfg["style"]:
            g = gram_matrix(feats[layer])
            acc[layer] = g if acc[layer] is None else acc[layer] + g

        count += 1

    for layer in acc:
        acc[layer] /= count

    print(f"[✓] Style grams averaged over {count} images.")
    return acc


def evaluate_one_model(model_name, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"  Evaluating: {model_name}")
  
    model_folder  = os.path.join(RUN_ROOT, model_name)
    test_root     = os.path.join(model_folder, "test")
    outputs_dir   = os.path.join(test_root, "outputs")
    metrics_dir   = os.path.join(test_root, "metrics")
    plots_dir     = os.path.join(test_root, "plots")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    ckpt_path = os.path.join(model_folder, "train", "best_checkpoint.pth")

    print(f"[✓] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    vgg, layer_cfg = load_vgg(device)
    style_grams = compute_style_grams(STYLE_PATH, device, vgg, layer_cfg)

  
    test_ds = MaskedImageDataset(TEST_PATH)
    loader = DataLoader(test_ds, batch_size=1)

    psnrs, ssims, grams = [], [], []

    for idx, (combined, rgb) in enumerate(loader):
        combined = combined.to(device)
        rgb      = rgb.to(device)

        with torch.no_grad():
            gen = model(combined)

        out_path = os.path.join(outputs_dir, f"output_{idx:04d}.png")
        T.ToPILImage()(gen.squeeze(0).cpu()).save(out_path)


        gen_feats = vgg(gen)
        rgb_feats = vgg(rgb)

        psnrs.append(psnr(gen, rgb).item())
        ssims.append(ssim(gen, rgb).item())
        grams.append(gram_distance(gen_feats, rgb_feats, layer_cfg["style"]).item())

    results = {
        "avg_psnr": sum(psnrs)/len(psnrs),
        "avg_ssim": sum(ssims)/len(ssims),
        "avg_gram": sum(grams)/len(grams),
        "per_image": {
            "psnr": psnrs,
            "ssim": ssims,
            "gram": grams
        }
    }

    save_json(os.path.join(metrics_dir, "test_metrics.json"), results)

    plot_curve(psnrs, "Test PSNR (per image)", "PSNR", os.path.join(plots_dir, "psnr_curve.png"))
    plot_curve(ssims, "Test SSIM (per image)", "SSIM", os.path.join(plots_dir, "ssim_curve.png"))
    plot_curve(grams, "Test Gram Distance (per image)", "Gram", os.path.join(plots_dir, "gram_curve.png"))

    print(f"[✓] Completed: {model_name}")
    print("Outputs:", outputs_dir)
    print("Metrics:", metrics_dir)
    print("Plots:", plots_dir)


if __name__ == "__main__":

    evaluate_one_model("baseline", TransformerNetBaseline())
    evaluate_one_model("stylized", TransformerNetStylized())