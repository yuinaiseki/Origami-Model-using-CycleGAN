""" Vanilla NST implementation """

# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------
import torch
from torchvision.models import vgg19
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import os

vgg = vgg19(pretrained=True).features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 2. MODEL DEFINITIONS 
# ------------------------------------------------------------
IMG_SIZE = 512                  # change to 256 for quick testing
LEARNING_RATE = 0.003
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e6
NUM_STEPS = 10

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

# ------------------------------------------------------------
# 3. CONFIGS
# ------------------------------------------------------------
"""
Different configurations we tried, with details/rationale for each one:
    gatys: Baseline - proven to work well, balanced across scales
    geometric_emphasis: Focus on mid-level layers (conv2-4) that capture geometric patterns and edges - ideal for origami's angular structures
    edge_heavy: Only early layers with high weights - maximizes sharp fold detection
    planar_surfaces: Mid-to-deep layers that capture flat regions - paper is planar
    equal_weights: Removes bias - lets all scales contribute equally
    high_detail_content: Earlier content layer preserves more detail - might keep animal features clearer
    minimal_fast: Quick baseline with fewer layers - good for testing
    texture_focused: Multiple early layers for paper texture
"""

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
        'description': 'Original Gatys - balanced multi-scale style transfer'
    },
        
    'geometric_emphasis': {
        'content': ['conv4_2'],
        'style': ['conv2_1', 'conv3_1', 'conv4_1'],  # Skip very fine and very coarse
        'style_weights': {
            'conv2_1': 1.5,  # Emphasize edges and angles
            'conv3_1': 1.5,  # Emphasize geometric patterns
            'conv4_1': 1.0   # Larger geometric structures
        },
        'description': 'Mid-level emphasis - strong geometric patterns and edges for origami'
    },
    
    'edge_heavy': {
        'content': ['conv4_2'],
        'style': ['conv1_1', 'conv2_1'],  # Only early layers
        'style_weights': {
            'conv1_1': 2.0,  # Very strong edge emphasis
            'conv2_1': 1.5   # Strong local patterns
        },
        'description': 'Early layers only - sharp folds and crisp edges'
    },
    
    'planar_surfaces': {
        'content': ['conv4_2'],
        'style': ['conv3_1', 'conv4_1'],  # Mid-to-deep layers
        'style_weights': {
            'conv3_1': 1.5,  # Flat surface patterns
            'conv4_1': 1.5   # Large planar regions
        },
        'description': 'Mid-deep layers - flat paper-like surfaces'
    },
    
    'equal_weights': {
        'content': ['conv4_2'],
        'style': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        'style_weights': {
            'conv1_1': 1.0,
            'conv2_1': 1.0,
            'conv3_1': 1.0,
            'conv4_1': 1.0,
            'conv5_1': 1.0
        },
        'description': 'Equal weights - no bias toward any scale'
    },
    
    'high_detail_content': {
        'content': ['conv3_1'],  # Earlier layer = more detail??
        'style': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
        'style_weights': {
            'conv1_1': 1.0,
            'conv2_1': 1.0,
            'conv3_1': 0.8,
            'conv4_1': 0.5
        },
        'description': 'Earlier content layer - preserves more fine details'
    },
    
    'minimal_fast': {
        'content': ['conv4_2'],
        'style': ['conv2_1', 'conv3_1'],  # Only 2 layers
        'style_weights': {
            'conv2_1': 1.0,
            'conv3_1': 1.0
        },
        'description': 'Minimal layers - faster computation, test baseline'
    },
    
    'texture_focused': {
        'content': ['conv4_2'],
        'style': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'],  # Early layers + alternate conv
        'style_weights': {
            'conv1_1': 1.2,
            'conv1_2': 1.0,
            'conv2_1': 1.2,
            'conv2_2': 1.0
        },
        'description': 'Multiple early layers - paper texture emphasis'
    }
}


ACTIVE_LAYER_CONFIG = "gatys" 

# ------------------------------------------------------------
# 4. UTILS
# ------------------------------------------------------------
def img_to_tensor(path, max_size=IMG_SIZE):
    """load image and preprocess it for VGG19"""

    img = Image.open(path).convert("RGB")

    # resize
    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size) 

    # resize and normalize image for vgg19
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # mean and std taken from ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_img = transform(img).unsqueeze(0)
    return processed_img.to(device)

def tensor_to_img(tensor):
    """convert image back to viewable from optimized tensor"""
    # get img, get rid of gradients
    img = tensor.cpu().clone().detach()
    img = img.squeeze(0)
    # de-normalize
    img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225])(img)
    img = img.clamp(0,1)
    img = transforms.ToPILImage()(img)
    return img


def extract_features(img_tensor, layers, model=vgg):
    """feature extraction"""

    x = img_tensor

    # initializing features dictionary
    features={}
    
    # dict for layer name -> index
    layers_to_extract = {LAYER_INDICES[layer]: layer for layer in layers}

    for index, layer in model._modules.items():
        x = layer(x)
        
        if index in layers_to_extract:
            features[layers_to_extract[index]] = x    
            
    return features

def save_layer(layer_tensor, layer_path, max_channels=16):
    """helper function to save one layer as a figure"""

    features = layer_tensor.squeeze(0).cpu().detach()
    num_channels = min(features.shape[0], max_channels)

    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels*2, 2))
    if num_channels == 1:
        axes = [axes]
    
    for ch in range(num_channels):
        feature_map = features[ch]
        
        # Normalize to [0, 1]
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[ch].imshow(feature_map, cmap='viridis')
        axes[ch].axis('off')
        axes[ch].set_title(f'Ch {ch}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(layer_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def save_all_layers(img_tensor, img_type, obj_name, output_dir, model=vgg):
    """saving all the layers"""

    os.makedirs(output_dir, exist_ok=True)

    # get list of all layers
    layers_list = list(LAYER_INDICES.keys())
    features = extract_features(img_tensor, layers_list)

    for layer in layers_list:
        layer_path = os.path.join(output_dir, f"{obj_name}_{img_type}_{layer}.png")
        save_layer(features[layer],layer_path)
    
    return None


def gram_matrix(img_tensor):
    """gram matrix to capture style"""
    batch, channels, height, width = img_tensor.size()
    # flatten
    img_tensor = img_tensor.view(channels, height * width)
    gram = torch.mm(img_tensor, img_tensor.t()) / (channels * height * width)
    return gram
    
def nst(content_path, style_path, obj_name, output_path=None,
        num_steps=NUM_STEPS, 
        style_weight=STYLE_WEIGHT, 
        content_weight=CONTENT_WEIGHT,
        alpha=LEARNING_RATE,
        config_name=ACTIVE_LAYER_CONFIG,
        metric_callback=None,
        output_dir=None,):
    """core NST implementation: logs numbers (e.g. loss, avg time taken) each step + image results in specified dir
    
    Args:
        content_path: Either a file path (str) or a PyTorch tensor
        style_path: Either a file path (str) or a PyTorch tensor
    """
    
    start_time = time.time()
    
    # get config
    config = LAYER_CONFIGS[config_name]
    content_layers = config["content"]
    style_layers = config["style"]
    style_layer_weights = config['style_weights']

    # model
    vgg_model = vgg19(pretrained=True).features
    for param in vgg_model.parameters():
        param.requires_grad_(False)
    vgg_model.to(device).eval()

    print("loading images")
    # imgs, in tensor format - handle both paths and tensors
    if isinstance(content_path, str):
        content_tensor = img_to_tensor(content_path)
    else:
        # Already a tensor - make sure it's on the right device
        content_tensor = content_path.to(device)
        
    if isinstance(style_path, str):
        style_tensor = img_to_tensor(style_path)
    else:
        # Already a tensor - make sure it's on the right device
        style_tensor = style_path.to(device)

    print("getting features")
    # extract
    content_features = extract_features(content_tensor, content_layers, model=vgg_model)
    style_features = extract_features(style_tensor, style_layers, model=vgg_model)

    print("calculating grams")
    # calculate gram
    style_grams = {layer: gram_matrix(style_features[layer]) 
                   for layer in style_layers}
    
    result = content_tensor.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([result], lr=alpha)

    loss_history = {
        'total': [],
        'content': [],
        'style': [],
        'step': []
    }

    if output_dir:
            final_output_dir = os.path.join(output_dir, f'{obj_name}')
            os.makedirs(final_output_dir, exist_ok=True)
            log_file = os.path.join(final_output_dir, f'training_log_{obj_name}.txt')
            with open(log_file, 'w') as f:
                f.write(f"NST training log\n")
                f.write(f"{'='*80}\n")
                f.write(f"config: {config_name}\n")
                f.write(f"content: {content_path if isinstance(content_path, str) else 'tensor'}\n")
                f.write(f"style: {style_path if isinstance(style_path, str) else 'tensor'}\n")
                f.write(f"total steps: {num_steps}\n")
                f.write(f"content weight: {content_weight}\n")
                f.write(f"style weight: {style_weight}\n")
                f.write(f"learning rate: {alpha}\n")
                f.write(f"{'='*80}\n\n")

            print(f"started logging to: {log_file}")

    start_time = time.time()
    last_log_time = start_time

    for step in range(num_steps):
        result_features = extract_features(result, content_layers+style_layers, model=vgg_model)

        # content
        content_loss = 0
        for layer in content_layers:
            content_loss += torch.mean((result_features[layer] - content_features[layer])**2)
        # avg loss for all layers
        content_loss = content_loss / len(content_layers)

        style_loss = 0
        for layer in style_layers:
            result_feature = result_features[layer]
            gram = gram_matrix(result_feature)
            style_gram = style_grams[layer]

            layer_weight = style_layer_weights.get(layer, 1.0)
            
            batch, channels, height, width = result_feature.shape
            layer_style_loss = layer_weight * torch.mean(
                (gram - style_gram)**2
            )
            style_loss += layer_style_loss / (channels * height * width)

        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history['total'].append(total_loss.item())
        loss_history['content'].append(content_loss.item())
        loss_history['style'].append(style_loss.item())
        loss_history['step'].append(step)
        
        if metric_callback:
            metric_callback(step, total_loss.item(), content_loss.item(), style_loss.item(), result)

        if step % 100 == 0:
            current_time = time.time()
            cumulative_time = current_time - start_time
            step_time = current_time - last_log_time
            
            # console
            print(f"step {step:4d}/{num_steps} | "
                f"total loss: {total_loss.item():10.2f} | "
                f"content loss: {content_loss.item():8.4f} | "
                f"style loss: {style_loss.item():10.6f} | "
                f"time taken (cumulative): {cumulative_time:6.2f}s | " 
                f"time taken (~100 steps): {step_time:6.2f}s")
            
            # Save image
            if output_dir:
                intm_img = tensor_to_img(result)
                intm_path = os.path.join(final_output_dir, f'{obj_name}_step_{step:04d}.png')
                intm_img.save(intm_path)
                
                # Append to log file
                with open(log_file, 'a') as f:
                    f.write(f"{step:<8} {total_loss.item():<15.2f} "
                        f"{content_loss.item():<15.4f} {style_loss.item():<15.6f} "
                        f"{step_time:<12.2f}\n")
                    
            last_log_time = current_time

    end_time = time.time()
    total_time = end_time - start_time

    final_img = tensor_to_img(result)
    if output_dir:
        final_path = os.path.join(final_output_dir, f'{obj_name}_final.png')
        final_img.save(final_path)
        print(f"saved final result in {final_path}")
        
        # Append final summary to log
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL RESULTS!!\n")
            f.write(f"{'='*80}\n")
            f.write(f"total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
            f.write(f"avg time per step: {total_time/num_steps:.3f}s\n")
            f.write(f"final total loss: {loss_history['total'][-1]:.2f}\n")
            f.write(f"final content loss: {loss_history['content'][-1]:.4f}\n")
            f.write(f"final style loss: {loss_history['style'][-1]:.6f}\n")
        
        print(f"training log saved in {log_file}")
    
    print(f"time taken: {total_time:.2f}s")
    
    return final_img

def compare_configs(content_path, style_path, obj_name, output_dir, 
                   configs_to_test=None, num_steps=2000):
    """tests all (or specified) configurations of NST, logs and saves results"""
    
    # If no configs specified, test all
    if configs_to_test is None:
        configs_to_test = list(LAYER_CONFIGS.keys())
    
    print(f"# CONFIGURATION COMPARISON: {obj_name}")
    print(f"Configs: {configs_to_test}")
    print(f"# steps: {num_steps}")
    
    results = {}
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, f'{obj_name}_config_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run each configuration
    for config_name in configs_to_test:
        print(f"Testing Configuration: {config_name}")
        print(f"Description: {LAYER_CONFIGS[config_name].get('description', 'No description')}")
        
        # Create subdirectory for this config
        config_output_dir = os.path.join(comparison_dir, config_name)
        
        start_time = time.time()
        
        # Run NST
        final_img = nst(
            content_path=content_path,
            style_path=style_path,
            obj_name=f"{obj_name}_{config_name}",
            num_steps=num_steps,
            config_name=config_name,
            output_dir=config_output_dir
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results[config_name] = {
            'image': final_img,
            'time': elapsed_time,
            'output_dir': config_output_dir
        }
        
        print(f"\n✓ {config_name} complete! Time: {elapsed_time:.2f}s\n")
    
    # Create comparison grids    
    num_configs = len(results)
    cols = 3  # 3 configs per row
    rows = (num_configs + cols - 1) // cols  # Ceiling div
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = axes.flatten() if num_configs > 1 else [axes]
    
    for idx, (config_name, data) in enumerate(results.items()):
        axes[idx].imshow(data['image'])
        axes[idx].set_title(
            f"{config_name}\n{data['time']:.1f}s", 
            fontsize=10, 
            fontweight='bold'
        )
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    comparison_grid_path = os.path.join(comparison_dir, f'{obj_name}_all_configs.png')
    plt.savefig(comparison_grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison grid saved: {comparison_grid_path}")
    
    # Create summary report
    summary_path = os.path.join(comparison_dir, f'{obj_name}_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Configuration Comparison Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Object: {obj_name}\n")
        f.write(f"Content: {content_path}\n")
        f.write(f"Style: {style_path}\n")
        f.write(f"Steps: {num_steps}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"Results:\n")
        f.write(f"{'='*80}\n\n")
        
        for config_name, data in results.items():
            config = LAYER_CONFIGS[config_name]
            f.write(f"Configuration: {config_name}\n")
            f.write(f"Description: {config.get('description', 'N/A')}\n")
            f.write(f"Time: {data['time']:.2f}s ({data['time']/60:.2f} min)\n")
            f.write(f"Content layers: {config['content']}\n")
            f.write(f"Style layers: {config['style']}\n")
            f.write(f"Style weights: {config['style_weights']}\n")
            f.write(f"Output: {data['output_dir']}\n")
            f.write(f"\n{'-'*80}\n\n")
    
    print(f"Summary saved to: {summary_path}")
    
    print(f"\n{'#'*80}")
    print(f"# ALL CONFIGURATIONS COMPLETE!")
    print(f"{'#'*80}")
    print(f"Results saved in: {comparison_dir}/")
    print(f"{'#'*80}\n")
    
    return results

# ------------------------------------------------------------
# 4. STANDALONE TESTING
# ------------------------------------------------------------
if __name__ == "__main__":

    content_path = "./test_imgs/cat.jpg"
    style_path = "./test_imgs/rose.jpg"
    output_path = "./results"
    obj_name = "cat"

    org_img = Image.open(content_path)

    result = nst(
            content_path=content_path,
            style_path=style_path,
            obj_name=obj_name,
            output_path=None,
            num_steps=2000,                   # more?
            config_name="gatys",
            output_dir=output_path
        )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(org_img)
    axes[1].imshow(result)

    plt.tight_layout()
    plt.show()
    
   