import json
import os
import re
import matplotlib.pyplot as plt
from nst import nst, LAYER_CONFIGS, vgg, extract_features
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

SAVE_DIR = "evaluate/vanilla"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device).eval()

# Image pairs to test
IMAGE_PAIRS = [
    {
        # original img pair
        "name": "butterfly",
        "content": "../test_imgs/butterfly.jpg",
        "style": "../test_imgs/butterfly_o.jpg"
    },
    {
        # matching position images, not segmented
        "name": "butterfly_matched",
        "content": "../test_imgs/butterfly2.jpg",
        "style": "../test_imgs/butterfly2_o_p.png"
    },
    {
        # matching position images, segmented
        "name": "butterfly_matched_segmented",
        "content": "../test_imgs/butterfly2_masked.png",
        "style": "../test_imgs/betterfly2_o_p.jpg"
    },
]

# Transform WITH normalization (for VGG19)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Denormalization transform (for visualization and metrics)
denorm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def load_image(path):
    """Load image and convert to RGB (handles RGBA)"""
    img = Image.open(path)
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def to_numpy(tensor):
    """Convert tensor to numpy array with proper denormalization"""
    # Denormalize first
    img = tensor.clone().detach()
    img = denorm(img.squeeze(0))
    # Clamp to [0, 1] range
    img = torch.clamp(img, 0, 1)
    # Convert to numpy
    return np.transpose(img.cpu().numpy(), (1, 2, 0))

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    F = tensor.view(c, h*w)
    return torch.mm(F, F.t())

# Configurations to test (from vgg_layers.py)
configs_to_test = [
    'gatys',
    'geometric_emphasis',
    'edge_heavy',
    'planar_surfaces',
    'equal_weights',
    'high_detail_content',
    'minimal_fast',
    'texture_focused'
]

def run_nst_with_metrics(config_name, content, style, image_name):
    metrics = {
        "step": [],
        "ssim": [],
        "psnr": [],
        "gram_distance": [],
        "total_loss": []
    }

    cfg = LAYER_CONFIGS[config_name]

    def metric_callback(step, total_loss, content_loss, style_loss, result):
        # Only record every 10 steps
        if step % 10 != 0:
            return
            
        result_np = to_numpy(result)
        content_np = to_numpy(content)
        
        # Ensure both images have the same dimensions
        min_height = min(result_np.shape[0], content_np.shape[0])
        min_width = min(result_np.shape[1], content_np.shape[1])
        result_np = result_np[:min_height, :min_width]
        content_np = content_np[:min_height, :min_width]
        
        ssim_val = ssim(content_np, result_np, channel_axis=2, data_range=1.0)
        psnr_val = psnr(content_np, result_np, data_range=1.0)
        
        # Calculate gram distance
        with torch.no_grad():
            result_feats = extract_features(result, cfg['style'])
        
        gram_dist = 0
        for layer in cfg["style"]:
            gram_result = gram_matrix(result_feats[layer])
            gram_dist += torch.norm(gram_result - style_grams[layer]).item()

        metrics["step"].append(step)
        metrics["ssim"].append(ssim_val)
        metrics["psnr"].append(psnr_val)
        metrics["gram_distance"].append(gram_dist)
        metrics["total_loss"].append(total_loss)

    # Run NST
    nst(content, style, f"{image_name}_{config_name}", num_steps=2000, 
        config_name=config_name, output_dir=SAVE_DIR, metric_callback=metric_callback)

    return metrics

def plot(metric_name, values, steps, config_name, image_name):
    plt.figure(figsize=(6,4))
    plt.plot(steps, values, marker="o", markersize=2, linewidth=1)  # Smaller markers
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Step - {image_name}")
    plt.grid(True)
    plt.tight_layout()

    fname = metric_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(SAVE_DIR, f"{fname}_{config_name}_{image_name}.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_DIR, f"{fname}_{config_name}_{image_name}.pdf"))
    plt.close()

if __name__ == "__main__":
    for image_pair in IMAGE_PAIRS:
        image_name = image_pair["name"]
        content_path = image_pair["content"]
        style_path = image_pair["style"]

        print(f"Processing image pair: {image_name}")

        # Load content and style images with RGBA handling
        content_img = load_image(content_path)
        style_img = load_image(style_path)
        
        content = transform(content_img).unsqueeze(0).to(device)
        style = transform(style_img).unsqueeze(0).to(device)

        for config_name in configs_to_test:
            print(f"Evaluating configuration: {config_name}")
            
            # Precompute style features
            cfg = LAYER_CONFIGS[config_name]
            with torch.no_grad():
                style_feats = extract_features(style, cfg['style'])
            style_grams = {
                layer: gram_matrix(style_feats[layer])
                for layer in cfg["style"]
            }
            
            metrics = run_nst_with_metrics(config_name, content, style, image_name)
            
            # Save metrics to JSON
            metrics_clean = {
                k: [float(v) for v in metrics[k]] if isinstance(metrics[k], list) else metrics[k]
                for k in metrics
            }
            with open(os.path.join(SAVE_DIR, f"metrics_vanilla_{config_name}_{image_name}.json"), "w") as f:
                json.dump(metrics_clean, f, indent=4)

            # Plot metrics
            for metric in ["ssim", "psnr", "gram_distance", "total_loss"]:
                plot(metric.replace("_", " ").title(), metrics[metric], metrics["step"], config_name, image_name)

    print(f"Metrics and plots saved in {SAVE_DIR}")