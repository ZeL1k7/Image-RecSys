import torch
from PIL import Image
from pathlib import Path
import os
from functools import lru_cache


class Embedder:
    def __init__(self, backbone):
        self._backbone = backbone

    def get_embeddings(self, image_batch) -> torch.Tensor:
        return self._backbone(image_batch)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, transforms):
        super().__init__()
        self.data_path = data_path
        self._dataset = [self.data_path + path for path in os.listdir(self.data_path)]
        self._transforms = transforms

    def __getitem__(self, idx):
        image_path = self._dataset[idx]
        image = Image.open(image_path).convert("RGB")
        image = self._transforms(image)
        return image, idx

    def __len__(self):
        return len(os.listdir(self.data_path))


@lru_cache(1)
def load_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model
