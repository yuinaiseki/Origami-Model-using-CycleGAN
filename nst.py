import torch
from torchvision.models import vgg19

# load pre-trained model and get weights
vgg = vgg19(pretrained=True).features

# load image and preprocess it for VGG19

# convert image back to viewable from optimized tensor

# feature extration 

# gram matrix to capture style

# style transfer

# running the whole thing: 
# if __name__ == "__main__"