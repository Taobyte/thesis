import pandas as pd
import numpy as np
import pickle
import os

from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy import signal
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
            heart_rate = data["label"]
            wrist_acc = data["signal"]["wrist"]["ACC"]
            acc_norm = np.linalg.norm(wrist_acc, axis=1)  # 32Hz
            acc_norm_ppg = signal.resample(acc_norm, len(bvp))  # 64Hz

            window_size = 256  # 8 seconds at 32Hz
            stride = 64  # 2 seconds at 32Hz

            windows = sliding_window_view(acc_norm, window_shape=window_size)[::stride]
            acc_norm_heart_rate = np.mean(windows, axis=1)

            assert heart_rate.shape == acc_norm_heart_rate.shape

            activity = data["activity"]
            np.savez(
                dalia_preprocessed_dir + "/" + (str(path).split("\\")[-2]),
                bvp=bvp,
                heart_rate=heart_rate,
                acc_norm_ppg=acc_norm_ppg,
                acc_norm_heart_rate=acc_norm_heart_rate,
                activity=activity,
            )

    print("Finished processing dalia files.")


def ucihar_preprocess(datadir: str):
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]
    train_val_subjects = np.loadtxt(datadir + "train/subject_train.txt")
    y_train = np.loadtxt(datadir + "train/y_train.txt")
    y_test = np.loadtxt(datadir + "test/y_test.txt")

    def load_signals(mode: str):
        assert mode in ["train", "test"]
        signals = []
        for sig in INPUT_SIGNAL_TYPES:
            signals.append(
                np.loadtxt(datadir + f"{mode}/Inertial Signals/{sig}{mode}.txt")[
                    :, :, np.newaxis
                ]
            )

        signals = np.concatenate(signals, axis=-1)
        return signals

    X_train = load_signals("train")
    X_test = load_signals("test")
    np.savez(
        datadir + "ucihar_preprocessed.npz",
        train_val_subjects=train_val_subjects,
        y_train=y_train,
        y_test=y_test,
        X_train=X_train,
        X_test=X_test,
    )


if __name__ == "__main__":
    # path = "C:/Users/cleme/ETH/Master/Thesis/data/Capture24/capture24/"
    # create_capture24_npy_files(path)
    datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
    create_dalia_npy_files(datadir)
    # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/WildPPG/data"
    # create_wildppg_npy_files(datadir)
    # datadir = (
    #      "C:/Users/cleme/ETH/Master/Thesis/data/UCIHAR/UCI HAR Dataset/UCI HAR Dataset/"
    # )
    # ucihar_preprocess(datadir)
