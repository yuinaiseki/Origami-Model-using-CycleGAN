import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class MaskedImageDataset(Dataset):
    def __init__(self, folder, image_size=256):
        self.folder = folder
        self.image_size = image_size

        self.image_files = []
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_mask" not in f:
                base = f.split(".")[0]
                mask_file = base + "_mask.png"
                mask_path = os.path.join(folder, mask_file)

                if os.path.exists(mask_path):
                    self.image_files.append(f)

        self.transform_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base = img_name.split(".")[0]
        img_path = os.path.join(self.folder, img_name)
        mask_path = os.path.join(self.folder, base + "_mask.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        img_t = self.transform_img(img)      
        mask_t = self.transform_mask(mask)    

        combined = torch.cat([img_t, mask_t], dim=0)  
        return combined, img_t 
