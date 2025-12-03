import torch
import torch.nn as nn
from torchvision import models


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
        'style': [
            'conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1'
        ],
        'style_weights': {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        },
    }
}

ACTIVE_LAYER_CONFIG = 'gatys'


class VGGFeatures(nn.Module):

    def __init__(self, vgg, layer_indices):
        super().__init__()
        self.vgg = vgg
        self.idx_to_name = {
            int(idx_str): name for name, idx_str in layer_indices.items()
        }

    def forward(self, x):
        feats = {}
        cur = x

        for idx, layer in self.vgg._modules.items():
            cur = layer(cur)
            i = int(idx)

            if i in self.idx_to_name:
                feats[self.idx_to_name[i]] = cur

        return feats


def load_vgg(device):

    vgg_model = models.vgg19(pretrained=True).features.to(device).eval()


    for p in vgg_model.parameters():
        p.requires_grad = False

    vgg_feats = VGGFeatures(vgg_model, LAYER_INDICES).to(device).eval()
    layer_cfg = LAYER_CONFIGS[ACTIVE_LAYER_CONFIG]

    return vgg_feats, layer_cfg
