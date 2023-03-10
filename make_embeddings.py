import argparse
import gc
from utils import load_model, ImageDataset
import torch
import torchvision
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings")
    parser.add_argument("dataset_path", type=str, help="data files")
    parser.add_argument("embeddings_path", type=str, help="embedding files")
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_list = []
    cnt, part = 0, 0

    resnet = torchvision.models.resnet101(pretrained=True)
    resnet.fc = torch.nn.Identity()
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = ImageDataset(args.dataset_path, data_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    embedder = load_model(resnet, DEVICE)

    with torch.no_grad():
        for batch in loader:
            img, idx = batch
            img = img.to(DEVICE)
            embeddings_list.append(
                embedder.get_embeddings(img).view(-1).cpu().detach().numpy()
            )
            del img
            cnt += 1
            if cnt % 50 == 0:
                part += 1
                np.save(
                    args.dataset_path + "embeddings_" + str(part) + ".npy",
                    embeddings_list,
                )
                del embeddings_list
                embeddings_list = []
            gc.collect()
        np.save(args.dataset_path + "embeddings_last.npy", embeddings_list)
