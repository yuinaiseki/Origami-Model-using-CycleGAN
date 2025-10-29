"""
Visualize each conv layer of VGG network to compare/contrast: exploration tool for deciding configurations

Directory: "./NST"

"""

from nst import *
import os
from PIL import Image

def explore_layers(content_path, style_path, obj_name, output_dir='./layers'):
    final_output_dir = os.path.join(output_dir, f'{obj_name}')
    os.makedirs(final_output_dir, exist_ok=True)

    content_tensor = img_to_tensor(content_path)
    style_tensor = img_to_tensor(style_path)

    save_all_layers(content_tensor, "content", obj_name, final_output_dir)
    save_all_layers(style_tensor, "style", obj_name, final_output_dir)
    return None

if __name__ == "__main__":
    obj_1 = "rose"
    content_path_1 = "test_imgs/rose.jpg"
    style_path_1 = "test_imgs/rose_o.jpg"
    output_path = "./results" 
    num_steps = 2000

    explore_layers(content_path_1, style_path_1, obj_1)
    nst_standalone(content_path_1, style_path_1, output_path, obj_1, num_steps)

    
    obj_2 = "butterfly"
    content_path_2 = "test_imgs/butterfly.jpg"
    style_path_2 = "test_imgs/butterfly_o.jpg"

    explore_layers(content_path_2, style_path_2, obj_2)
    nst_standalone(content_path_2, style_path_2, output_path, obj_2, num_steps)

    obj_3 = "cat"
    content_path_3 = "test_imgs/cat.jpg"
    style_path_3 = "test_imgs/cat_o.jpg"

    explore_layers(content_path_3, style_path_3, obj_3)
    nst_standalone(content_path_3, style_path_3, output_path, obj_3, num_steps)

    obj_4 = "bird"
    content_path_4 = "test_imgs/bird.jpg"
    style_path_4 = "test_imgs/bird_o.jpg"

    explore_layers(content_path_4, style_path_4, obj_4)
    nst_standalone(content_path_4, style_path_4, output_path, obj_4, num_steps)




