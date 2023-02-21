import numpy as np
import pandas as pd
import os
from utils import ImageDataset

if __name__ == "__main__":
    TOP_K = 6
    rec_idxs = np.load(os.listdir("/kaggle/input/sumbission")[0], allow_pickle=True)
    for path in os.listdir("/kaggle/input/sumbission")[1:]:
        rec_idxs_curr = np.load(path, allow_pickle=True)
        rec_idxs = np.vstack((rec_idxs, rec_idxs_curr))
    dataset = ImageDataset("/kaggle/input/yandex-e/dataset/", None)
    df = pd.DataFrame(rec_idxs, columns=["filename", "ranking"])
    df.filename = df.filename.apply(lambda x: dataset.__getitem__(x[0]).split("/")[-1])
    df.ranking = df.ranking.apply(
        lambda x: " ".join(
            [dataset.__getitem__(i).split("/")[-1] for i in x[1 : 1 + TOP_K]]
        )
    )
    df = df.sort_values(by="filename")
    df = df.reset_index(drop=True)
    df.to_csv("submission.csv")
