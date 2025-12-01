""" Visualizing + logging each conv layer & configuration of VGG network to compare/contrast/explore configurations"""

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

# running configuration experiments on all test images
if __name__ == "__main__":

    # images to test
    objects = [
        ("rose", "test_imgs/rose.jpg", "test_imgs/rose_o.jpg"),
        ("butterfly", "test_imgs/butterfly.jpg", "test_imgs/butterfly_o.jpg"),
        ("cat", "test_imgs/cat.jpg", "test_imgs/cat_o.jpg"),
        ("bird", "test_imgs/bird.jpg", "test_imgs/bird_o.jpg"),
    ]
    
    output_path = "./results"
    num_steps = 2000  
    
    # configurations to test
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

    for obj_name, content_path, style_path in objects:
        compare_configs(
            content_path=content_path,
            style_path=style_path,
            obj_name=obj_name,
            output_dir=output_path,
            configs_to_test=configs_to_test,
            num_steps=num_steps
        )

 