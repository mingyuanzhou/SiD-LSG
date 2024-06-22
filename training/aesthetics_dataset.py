import math
import os
import random
import torch

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        path,
        resolution,
        random_crop=False,
        random_flip=0.0,
        prompt_only=False,
    ):
        super().__init__()

        self.name = 'aesthetics'
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.prompt_only = prompt_only
        assert self.prompt_only, 'prompt_only must set to True for Lion_aesthetics dataset.'
        self.prompt_list = []
        
        
        filenames = ['aesthetics_6_plus.txt', 'aesthetics_625_plus.txt', 'aesthetics_65_plus.txt']
        # Check if any of the preferred files exists in the specified path
        for filename in filenames:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                break
        with open(full_path, 'rt') as f:
            print(full_path)
            for row in f:
                ##
                row = row.strip('\n')
                ##
                self.prompt_list.append(row)

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return torch.zeros(1, 4, 4), self.prompt_list[idx]  # Return a dummy image when prompt_only is set.