from utils import load_model, ImageDataset
import torch
import torchvision
import gc
import numpy as np

if __name__ == "__main__":
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = torchvision.models.resnet101(pretrained=True)
    resnet.classifier.fc = torch.nn.Identity()
    dataset = ImageDataset("data/dataset/", data_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    embedder = load_model(resnet, device)
    embeddings_list = []
    cnt, part = 0, 0
    with torch.no_grad():
        for batch in loader:
            img, idx = batch
            img = batch.to(device)
            embeddings_list.append(
                embedder.get_embeddings(img).view(-1).cpu().detach().numpy()
            )
            del img
            cnt += 1
            if cnt % 3000:
                part += 1
                np.save("embeddings_" + str(part) + ".npy", embeddings_list)
                del embeddings_list
                embeddings_list = []
            gc.collect()
        np.save("embeddings_last.npy", embeddings_list)
