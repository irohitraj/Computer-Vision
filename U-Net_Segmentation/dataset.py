import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import torch.nn.functional as F
import time



class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, img_dir, image_size=(512, 512), transform=None, target_transform=None):
        self.coco = coco
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ids = list(sorted(coco.imgs.keys()))
        self.image_size = image_size
        
        # Default transforms if none provided
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Load and combine masks
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann:
                ann_mask = coco.annToMask(ann)
                mask = np.maximum(mask, ann_mask)
        
        # Convert to PIL Image for transforms
        mask = Image.fromarray(mask * 255)  # Scale to 0-255
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = self.image_transforms(img)
            
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = self.mask_transforms(mask)
        
        return img, mask.long()

    def __len__(self):
        return len(self.ids)
