import torch
from torchvision.models import vgg19
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# CONFIG
IMG_SIZE = 512 # change to 256 for quick testing

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

# gram matrix to capture style

# style transfer
# def nst():

# running the whole thing: 
if __name__ == "__main__":
    img_url = "./test_imgs/cat.jpg"

    org_img = Image.open(img_url)

    processed_img = img_to_tensor(img_url)
    converted_img = tensor_to_img(processed_img)
    print(converted_img.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(org_img)
    axes[1].imshow(converted_img)

    plt.tight_layout()
    plt.show()