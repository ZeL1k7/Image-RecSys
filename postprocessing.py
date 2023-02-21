import argparse
import os
import numpy as np
import pandas as pd
from utils import ImageDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_path", type=str, help="data files")
    parser.add_argument("candidates_path", type=str, help="candidates files")
    parser.add_argument("output_path", type=str, help="data files")
    args = parser.parse_args()
    TOP_K = 6
    rec_idxs = np.load(
        args.candidates_path + os.listdir(args.candidates_path)[0], allow_pickle=True
    )

    for path in os.listdir(args.candidates_path)[1:]:
        rec_idxs_curr = np.load(args.candidates_path + path, allow_pickle=True)
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
    df.to_csv(args.candidates_path + "submission.csv")
