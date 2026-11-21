import os
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UIEBDataset(Dataset):
    def __init__(self, raw_dir, img_size=(256, 256)):
        self.raw_dir = raw_dir
        self.img_paths = sorted(glob.glob(os.path.join(raw_dir, "*.png")) + glob.glob(os.path.join(raw_dir, "*.jpg")))
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def apply_random_candidate(self, pil_img):
        # Unsupervised Random Candidate Mechanism (Contrast, Saturation, Gamma)
        choice = random.choice(['contrast', 'saturation', 'gamma'])
        if choice == 'contrast':
            factor = random.uniform(1.2, 1.8)
            enhancer = ImageEnhance.Contrast(pil_img)
            return enhancer.enhance(factor)
        elif choice == 'saturation':
            factor = random.uniform(1.3, 2.0)
            enhancer = ImageEnhance.Color(pil_img)
            return enhancer.enhance(factor)
        else: # Gamma correction
            gamma = random.uniform(0.5, 0.8)
            img_np = np.array(pil_img) / 255.0
            img_gamma = np.power(img_np, gamma) * 255.0
            return Image.fromarray(img_gamma.astype(np.uint8))

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        raw_img = Image.open(path).convert('RGB')
        
        # Original Image & Random Candidate Transformation for Unsupervised Posterior Learning
        cand_img = self.apply_random_candidate(raw_img)
        
        raw_tensor = self.transform(raw_img)
        cand_tensor = self.transform(cand_img)
        
        return raw_tensor, cand_tensor