import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class EmotionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = int(self.data.iloc[idx, 1])
        image = np.load(npy_path)  # shape: (H, W), float32 in [0, 1]

        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, label
