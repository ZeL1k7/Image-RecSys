import argparse
import os
import gc
import numpy as np
from utils import ImageDataset, load_model
import torch
import torchvision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get candidates")
    parser.add_argument("dataset_path", type=str, help="data raw files")
    parser.add_argument("embeddings_path", type=str, help="embedding files")
    parser.add_argument("output_path", type=str, help="output files")
    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_idxs = {
        "embeddings_last.npy": 9000,
        "embeddings_2.npy": 6000,
        "embeddings_1.npy": 3000,
        "embeddings_0.npy": 0,
    }
    part = 0
    cnt = 0
    TOP_K = 10
    candidates = []

    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    resnet = torchvision.models.resnet101(pretrained=True)
    resnet.fc = torch.nn.Identity()
    dataset = ImageDataset(args.dataset_path, data_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    embedder = load_model(resnet, DEVICE)

    with torch.no_grad():
        for batch in loader:
            batch, idx = batch
            item_candidates = []
            idx_candidates = []
            batch = batch.to(DEVICE)
            user = embedder.get_embeddings(batch)
            bs = user.size(0)
            user = user.view(bs, -1)  # B x Embedding_dim
            for embedding_file in os.listdir(args.embeddings_path):
                item_embeddings = np.load(args.embeddings_path + embedding_file)
                item_embeddings = torch.from_numpy(item_embeddings).to(DEVICE)

                ranks = torch.nn.functional.cosine_similarity(
                    user, item_embeddings, dim=1
                )  # C
                idxs = torch.argsort(ranks, dim=0, descending=True)
                item_candidates.append(ranks[idxs[:TOP_K]].cpu().detach().numpy())
                idxs += embedding_idxs[embedding_file]
                idx_candidates.append(idxs[:TOP_K].cpu().numpy())
                del item_embeddings
                del idxs

            item_candidates = torch.tensor(np.array(item_candidates))
            idx_candidates = torch.tensor(
                np.array(idx_candidates)
            )  # Embedding_file x Candidates
            item_dims = item_candidates.size()  # Embedding_file x Candidates
            item_candidates = item_candidates.view(-1)
            idx_candidates = idx_candidates.view(-1)
            cand_idx_ranked = torch.argsort(item_candidates, dim=0, descending=True)
            idx_candidates = idx_candidates[cand_idx_ranked]
            idx_candidates = idx_candidates.view(
                item_dims
            )  # Embedding_file x Candidates
            candidates.append(
                [idx.cpu().numpy(), idx_candidates[0].cpu().numpy()]
            )  # Pick TOP_K

            cnt += 1
            print(cnt)
            if cnt % 1 == 0:
                np.save(
                    args.output_path + "candidates" + str(part) + ".npy",
                    np.array(candidates),
                )
                part += 1
                del candidates
                candidates = []
            gc.collect()
        np.save(args.output_path + "candidates_last.npy", np.array(candidates))
