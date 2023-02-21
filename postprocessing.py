import os
import numpy as np
import pandas as pd
from utils import ImageDataset


if __name__ == "__main__":
    TOP_K = 6
    rec_idxs = np.load(
        "data/processed/" + os.listdir("data/processed")[0], allow_pickle=True
    )

    for path in os.listdir("data/processed")[1:]:
        rec_idxs_curr = np.load("data/processed/" + path, allow_pickle=True)
        rec_idxs = np.vstack((rec_idxs, rec_idxs_curr))

    dataset = ImageDataset("data/raw/", None)
    df = pd.DataFrame(rec_idxs, columns=["filename", "ranking"])

    df.filename = df.filename.apply(
        lambda x: dataset.get_image_path(x[0]).split("/")[-1]
    )
    df.ranking = df.ranking.apply(
        lambda x: " ".join(
            [dataset.get_image_path(i).split("/")[-1] for i in x[1 : 1 + TOP_K]]
        )
    )

    df = df.sort_values(by="filename")
    df = df.reset_index(drop=True)
    df.to_csv("data/processed/submission.csv")
