import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm


def create_capture24_npy_files(datadir: str):
    csv_files = Path(datadir).glob("P*.csv")
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)
        df["annotation"] = df["annotation"].str.split().str[-1].astype(float)
        df = df.drop(["time"], axis=1)
        np.save(str(csv_file).split(".")[0], df.to_numpy())


def stratified_participant_split(
    df: pd.DataFrame, test_size=0.2, val_size=0.25, seed=42
):
    df["strata"] = df["age"].astype(str) + "_" + df["sex"].astype(str)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["strata"], random_state=seed
    )

    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["strata"], random_state=seed
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


if __name__ == "__main__":
    path = "C:/Users/cleme/ETH/Master/Thesis/data/Capture24/capture24/"
    # create_capture24_npy_files(path)
    df = pd.read_csv(path + "metadata.csv")  # your original participant table
    train_df, val_df, test_df = stratified_participant_split(df)

    print("Train:", train_df["pid"])
    print("Val:", val_df["pid"])
    print("Test:", test_df["pid"])
