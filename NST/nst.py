import torch
from torchvision.models import vgg19
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import os


# CONFIG
IMG_SIZE = 512 # change to 256 for quick testing
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

# select active layer config to change layers for feature extraction
ACTIVE_LAYER_CONFIG = "gatys" 

# load pre-trained model and get weights
vgg = vgg19(pretrained=True).features

# cpu or gpu?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load image and preprocess it for VGG19
def img_to_tensor(path, max_size=IMG_SIZE):
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


# convert image back to viewable from optimized tensor
def tensor_to_img(tensor):
    # get img, get rid of gradients
    img = tensor.cpu().clone().detach()
    img = img.squeeze(0)
    # de-normalize
    img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225])(img)
    img = img.clamp(0,1)
    img = transforms.ToPILImage()(img)
    return img

# feature extration 

def extract_features(img_tensor, layers, model=vgg):

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
    # normalize output
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
    
def save_all_layers(img_tensor, img_type, output_dir, model=vgg):
    os.makedirs(output_dir, exist_ok=True)

    # get list of all layers
    layers_list = list(LAYER_INDICES.keys())
    features = extract_features(img_tensor, layers_list)

    for layer in layers_list:
        layer_path = os.path.join(output_dir, f"{img_type}_{layer}.png")
        save_layer(features[layer],layer_path)
    
    return None

# gram matrix to capture style
def gram_matrix(img_tensor):
    batch, channels, height, width = img_tensor.size()
    # flatten
    img_tensor = img_tensor.view(channels, height * width)
    gram = torch.mm(img_tensor, img_tensor.t())
    return gram



# style transfer
def nst(content_path, style_path, obj_name, output_path=None,
        num_steps=NUM_STEPS, 
        style_weight=STYLE_WEIGHT, 
        content_weight=CONTENT_WEIGHT,
        alpha=LEARNING_RATE,
        config_name=ACTIVE_LAYER_CONFIG,
        output_dir=None,):
    
    start_time = time.time()
    
    # get config
    config = LAYER_CONFIGS[config_name]
    content_layers = config["content"]
    style_layers = config["style"]
    style_layer_weights = config['style_weights']

    # model
    vgg = vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg.to(device).eval()

    print("loading images")
    # imgs, in tensor format
    content_tensor = img_to_tensor(content_path)
    style_tensor = img_to_tensor(style_path)

    print("getting features")
    # extract
    content_features = extract_features(content_tensor, content_layers, model=vgg)
    style_features = extract_features(style_tensor, style_layers, model=vgg)

    print("calculating grams")
    # calculate gram for layers in style image
    style_grams = {layer: gram_matrix(style_features[layer]) 
                   for layer in style_layers}
    
    result = content_tensor.clone().requires_grad_(True).to(device)          # change to style_tensor for gatys faithulness?
    optimizer = optim.Adam([result], lr=alpha)

    loss_history = {
        'total': [],
        'content': [],
        'style': [],
        'step': []
    }

    if output_dir:
            log_file = os.path.join(output_dir, f'training_log_{obj_name}.txt')
            with open(log_file, 'w') as f:
                f.write(f"NST training log\n")
                f.write(f"{'='*80}\n")
                f.write(f"config: {config_name}\n")
                f.write(f"content: {content_path}\n")
                f.write(f"style: {style_path}\n")
                f.write(f"total steps: {num_steps}\n")
                f.write(f"content weight: {content_weight}\n")
                f.write(f"style weight: {style_weight}\n")
                f.write(f"learning rate: {alpha}\n")
                f.write(f"{'='*80}\n\n")

            print(f"started logging to: {log_file}")

    start_time = time.time()
    last_log_time = start_time

    for step in range(num_steps):
        result_features = extract_features(result, content_layers+style_layers, model=vgg)

        # content
        content_loss = 0
        for layer in content_layers:
            content_loss += torch.mean((result_features[layer] - content_features[layer])**2) # change?
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

        if step % 100 == 0:
            current_time = time.time()
            cumulative_time = current_time - start_time
            step_time = current_time - last_log_time
            
            # consol
            print(f"step {step:4d}/{num_steps} | "
                f"total loss: {total_loss.item():10.2f} | "
                f"content loss: {content_loss.item():8.4f} | "
                f"style loss: {style_loss.item():10.6f} | "
                f"time taken (cumulative): {cumulative_time:6.2f}s | " 
                f"time taken (~100 steps): {step_time:6.2f}s")
            
            # Save image
            if output_dir:
                intm_img = tensor_to_img(result)
                intm_path = os.path.join(output_dir, f'step_{step:04d}.png')
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
        final_path = os.path.join(output_dir, 'final.png')
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
    
    print(f"time taken us {total_time:.2f}")
    
    return final_img

def nst_standalone(content_path, style_path, output_dir, obj_name, num_steps):

    org_img = Image.open(content_path)

    result = nst(
            content_path=content_path,
            style_path=style_path,
            obj_name=obj_name,
            num_steps=num_steps,     
            output_path=None,              # more?
            config_name="gatys",
            output_dir=output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(org_img)
    axes[1].imshow(result)

    plt.tight_layout()
    plt.show()

# for standalone testing
if __name__ == "__main__":

    content_path = "./test_imgs/cat.jpg"
    style_path = "./test_imgs/rose.jpg"
    output_path = "./results/"
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


    """
    img_url = "./test_imgs/cat.jpg"

    org_img = Image.open(img_url)

    processed_img = img_to_tensor(img_url)
    converted_img = tensor_to_img(processed_img)
    print(converted_img.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(org_img)
    axes[1].imshow(converted_img)

    plt.tight_layout()
    plt.show()"""