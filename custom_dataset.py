import h5py
from PIL import Image

import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None):
        with h5py.File(data_path, 'r') as f:
            self.data, self.targets = f['data'][:], f['labels'][:]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
