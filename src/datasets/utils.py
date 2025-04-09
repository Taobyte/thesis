import pandas as pd
import numpy as np
import pickle
import os

from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm


def create_wildppg_npy_files(datadir: str):
    wildppg_preprocessed = os.path.join(datadir, "wildppg_preprocessed")
    os.makedirs(wildppg_preprocessed, exist_ok=True)

    for mat_file in tqdm(Path(datadir).glob("*.mat")):
        data = loadmat(mat_file)
        acc_x = data["wrist"]["acc_x"][0][0][0]["v"][0].T
        acc_y = data["wrist"]["acc_y"][0][0][0]["v"][0].T
        acc_z = data["wrist"]["acc_z"][0][0][0]["v"][0].T
        activity = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        ecg = data["sternum"]["ecg"][0][0][0]["v"][0].T
        ppg = data["wrist"]["ppg_g"][0][0][0]["v"][0].T
        np.savez(
            wildppg_preprocessed + "/" + str(mat_file).split("\\")[-1].split(".")[0],
            ecg=ecg,
            ppg=ppg,
            activity=activity,
        )


def create_capture24_npy_files(datadir: str):
    capture24_preprocessed = os.path.join(datadir, "capture24_preprocessed")
    os.makedirs(capture24_preprocessed, exist_ok=True)
    csv_files = Path(datadir).glob("P*.csv")

    print("Start processing capture24 files.")
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)
        df["annotation"] = df["annotation"].str.split().str[-1].astype(float)
        df = df.drop(["time"], axis=1)
        np.save(
            capture24_preprocessed + "/" + str(csv_file).split(".")[0].split("\\")[-1],
            df.to_numpy(),
        )

    print("Finished processing capture24 files.")


def create_dalia_npy_files(datadir: str):
    dalia_preprocessed_dir = os.path.join(datadir, "dalia_preprocessed")
    os.makedirs(dalia_preprocessed_dir, exist_ok=True)

    print("Start processing dalia files.")

    for path in tqdm(Path(datadir).glob("**/S*.pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            bvp = data["signal"]["wrist"]["BVP"]
            ecg = data["signal"]["chest"]["ECG"]
            wrist_acc = data["signal"]["wrist"]["ACC"]
            chest_acc = data["signal"]["chest"]["ACC"]
            activity = data["activity"]
            np.savez(
                dalia_preprocessed_dir + "/" + (str(path).split("\\")[-2]),
                bvp=bvp,
                ecg=ecg,
                wrist_acc=wrist_acc,
                chest_acc=chest_acc,
                activity=activity,
            )

    print("Finished processing dalia files.")


if __name__ == "__main__":
    # path = "C:/Users/cleme/ETH/Master/Thesis/data/Capture24/capture24/"
    # create_capture24_npy_files(path)
    # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
    # create_dalia_npy_files(datadir)
    datadir = "C:/Users/cleme/ETH/Master/Thesis/data/WildPPG/data"
    create_wildppg_npy_files(datadir)
