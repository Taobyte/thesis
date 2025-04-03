import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm


def create_capture24_npy_files(datadir: str):
    csv_files = Path(datadir).glob("P*.csv")
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)
        df["annotation"] = df["annotation"].str.split().str[-1].astype(float)
        df = df.drop(["time"], axis=1)
        np.save(str(csv_file).split(".")[0], df.to_numpy())


if __name__ == "__main__":
    path = "C:/Users/cleme/ETH/Master/Thesis/data/Capture24/capture24/"
    create_capture24_npy_files(path)
