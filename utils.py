import torch
from PIL import Image
from pathlib import Path
import os
from functools import lru_cache
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc


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


def visualize_recs(
    idx: int,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    embedder: torch.nn.Module,
    embeddings_idxs: Dict[Path, int],
    top_k: int = 6,
):
    item, idx = dataset.__getitem__(idx)
    item = item.unsqueeze(0).to(device)
    i = Image.open(dataset._dataset[idx])
    plt.figure(figsize=(5, 2))
    plt.imshow(i)
    plt.title("Main image")
    plt.show()
    candidates = []
    user = embedder.get_embeddings(item)
    bs = user.size(0)
    user = user.view(bs, -1)  # B x Embedding_dim
    item_candidates = []
    idx_candidates = []

    for embedding_file in os.listdir("/kaggle/input/yandex-embeddings"):
        item_embeddings = np.load("/kaggle/input/yandex-embeddings/" + embedding_file)
        item_embeddings = torch.from_numpy(item_embeddings).to(device)
        ranks = torch.nn.functional.cosine_similarity(user, item_embeddings, dim=1)  # C
        idxs = torch.argsort(ranks, dim=0, descending=True)
        item_candidates.append(ranks[idxs[:top_k]].cpu().detach().numpy())
        idxs += embeddings_idxs[embedding_file]
        idx_candidates.append(idxs[:top_k].cpu().numpy())
        del item_embeddings
        gc.collect()
