import torch
from torchvision.models import vgg19
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import time


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
    

# gram matrix to capture style
def gram_matrix(img_tensor):
    batch, channels, height, width = img_tensor.size()
    # flatten
    img_tensor = img_tensor.view(channels, height * width)
    gram = torch.mm(img_tensor, img_tensor.t())
    return gram


# style transfer
def nst(content_path, style_path, output_path=None,
        num_steps=NUM_STEPS, 
        style_weight=STYLE_WEIGHT, 
        content_weight=CONTENT_WEIGHT,
        alpha=LEARNING_RATE,
        config_name=ACTIVE_LAYER_CONFIG):
    
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

    for step in range(num_steps):
        step_start = time.time()
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

        if step % 100 == 0:
            step_end = time.time()
            step_total = step_end - step_start
            print(f"step {step}/{num_steps}")
            print(f"time taken for 100 steps: {step_total:.2f}")
            

    result = tensor_to_img(result)

    if output_path:
        result.save(output_path)
        print(f"Saved result to {output_path}")

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"time taken: {total_time:.2f}")
    
    return result

# running the whole thing: 
if __name__ == "__main__":

    content_path = "./test_imgs/cat.jpg"
    style_path = "./test_imgs/rose.jpg"

    org_img = Image.open(content_path)

    result = nst(
            content_path=content_path,
            style_path=style_path,
            output_path=None,
            num_steps=2000,                   # more?
            config_name="gatys"
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