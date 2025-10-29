"""
Visualize each conv layer of VGG network to compare/contrast: exploration tool for deciding configurations

"""

from nst import *
import os
from PIL import Image

def explore_layers(content_path, style_path, output_dir='./layers'):
    os.makedirs(output_dir, exist_ok=True)

    content_tensor = img_to_tensor(content_path)
    style_tensor = img_to_tensor(style_path)

    save_all_layers(content_tensor, "content", output_dir)
    save_all_layers(style_tensor, "style", output_dir)
    return None

if __name__ == "__main__":
    obj_1 = "rose"
    content_path_1 = "NST/test_imgs/rose.jpg"
    style_path_1 = "NST/test_imgs/rose_o.jpg"
    output_path = "./results"
    num_steps = 3

    explore_layers(content_path_1, style_path_1)
    nst_standalone(content_path_1, style_path_1, output_path, obj_1, num_steps)

    """
    obj_2 = "butterfly"
    content_path_2 = "NST/test_imgs/butterfly.jpg"
    style_path_2 = "NST/test_imgs/butterfly_o.jpg"

    explore_layers(content_path_2, style_path_2)
    nst_standalone(content_path_2, style_path_2, output_path, obj_2, num_steps)"""




