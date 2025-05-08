import pandas as pd
import numpy as np
import pickle
import argparse
import os
import gzip
import shutil
import random
import torch
import lightning as L

from einops import rearrange
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        name: str,
        freq: int,
        look_back_window: int,
        prediction_window: int,
        use_activity_info: bool,
    ):
        super().__init__()

        self.name = name
        self.freq = freq
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.use_activity_info = use_activity_info

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # used for Gaussian Process model

    def get_inducing_points(self, num_inducing: int = 500) -> torch.Tensor:
        # Ensure the train dataset is ready
        self.setup(stage="fit")

        assert self.train_dataset is not None, "Train dataset is not initialized."

        dataset_length = len(self.train_dataset)
        assert dataset_length >= num_inducing, (
            f"Cannot sample {num_inducing} inducing points from dataset of size {dataset_length}"
        )

        indices = random.sample(range(dataset_length), num_inducing)

        inducing_points = [self.train_dataset[i][0] for i in indices]

        inducing_points_tensor = torch.stack(inducing_points, dim=0)
        inducing_points_tensor = rearrange(inducing_points, "B T C -> B (T C)")

        print(inducing_points_tensor.shape)

        return inducing_points_tensor

    def get_train_dataset_length(self):
        self.setup(stage="fit")

        assert self.train_dataset is not None, "Train dataset is not initialized."

        print(
            f"Length of train dataset for Gaussian Process: {len(self.train_dataset)}"
        )

        return len(self.train_dataset)

    def _get_dataset(self, mode: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        return z-normalized dataset
        """
        self.setup(stage="fit")
        if mode == "train":
            dataloader = self.train_dataloader()
        elif mode == "val":
            dataloader = self.val_dataloader()
        else:
            raise NotImplementedError()

        from src.models.utils import local_z_norm

        lbws = []
        pws = []
        for look_back_window, prediction_window in dataloader:
            look_back_window, mean, std = local_z_norm(look_back_window)
            prediction_window, _, _ = local_z_norm(prediction_window, mean, std)
            look_back_window = look_back_window.detach().cpu().numpy()
            prediction_window = prediction_window.detach().cpu().numpy()

            lbws.append(look_back_window)
            pws.append(prediction_window)

        lbws_dataset = np.concatenate(lbws, axis=0)
        pws_dataset = np.concatenate(pws, axis=0)

        lbw_gb = lbws_dataset.nbytes / (1024**3)
        pw_gb = pws_dataset.nbytes / (1024**3)
        total_gb = lbw_gb + pw_gb

        print(f"[{mode}] look_back_window: {lbws_dataset.shape}, {lbw_gb:.4f} GB")
        print(f"[{mode}] prediction_window: {pws_dataset.shape}, {pw_gb:.4f} GB")
        print(f"[{mode}] total size: {total_gb:.4f} GB")

        return lbws_dataset, pws_dataset

    def get_train_dataset(self):
        return self._get_dataset("train")

    def get_val_dataset(self):
        return self._get_dataset("val")


def create_ieee_npz_files(datadir: str):
    ieee_preprocessed = os.path.join(datadir, "ieee_preprocessed")
    os.makedirs(ieee_preprocessed, exist_ok=True)
    signal_files = Path(datadir).glob("*[12].mat")
    bpm_files = Path(datadir).glob("*_BPMtrace.mat")

    fs = 125  # sampling rate
    window_duration = 8
    overlap_duration = 6

    from scipy.signal import butter, filtfilt
    from sklearn.preprocessing import MinMaxScaler

    def filter_butter(x, fs):
        f1 = 0.5
        f2 = 4
        Wn = [f1, f2]
        N = 4
        b, a = butter(N, Wn, btype="bandpass", fs=fs)
        filtered = filtfilt(b, a, x)
        # Normalize to range [0, 1]
        scaler = MinMaxScaler()
        filtered = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
        return filtered

    def preprocess_signal(signal: np.ndarray):
        # create windows
        windows = sliding_window_view(signal, window_shape=window_duration * fs)[
            :: (window_duration - overlap_duration) * fs
        ]
        # downsample from 125Hz => 25Hz
        downsampled_windows = windows[:, ::5]

        return downsampled_windows

    print("Start processing IEEE files...")
    for i, (signal_file, bpm_file) in enumerate(zip(signal_files, bpm_files)):
        signals = loadmat(signal_file)["sig"]
        bpm = loadmat(bpm_file)["BPM0"]

        ppg1 = filter_butter(signals[1], fs)
        ppg2 = filter_butter(signals[2], fs)
        acc_x = filter_butter(signals[3], fs)
        acc_y = filter_butter(signals[4], fs)
        acc_z = filter_butter(signals[5], fs)

        avg_ppg = (ppg1 + ppg2) / 2
        acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        ppg = preprocess_signal(avg_ppg)[:, :, np.newaxis]  # add concatenation axis
        acc = preprocess_signal(acc)[:, :, np.newaxis]

        ppg_final = ppg[:-1]
        acc_final = acc[:-1]
        if len(ppg_final) != len(bpm):
            ppg_final = ppg[:]
            acc_final = acc[:]

        assert len(ppg_final) == len(acc_final), (
            f"signal_length: {len(ppg)} | bpm_length: {len(bpm)}"
        )

        assert len(ppg) == len(bpm), (
            f"signal_length: {len(ppg)} | bpm_length: {len(bpm)}"
        )

        np.savez(
            ieee_preprocessed + "/" + f"IEEE_{i}",
            ppg=ppg_final,
            acc=acc_final,
            bpms=bpm,
        )

    print("End processing IEEE files.")


def create_capture24_npy_files(datadir: str):
    capture24_preprocessed = os.path.join(datadir, "capture24_preprocessed")
    os.makedirs(capture24_preprocessed, exist_ok=True)

    zip_files = Path(datadir).glob("P*.csv.gz")

    print("Start processing capture24 files.")
    for zip_file in tqdm(zip_files):
        name = zip_file.stem  # Returns 'P023.csv' (but we want 'P023')
        name = zip_file.stem.split(".")[0]  # Returns 'P023'
        csv_file_path = datadir + name + ".csv"
        with gzip.open(f"{csv_file_path}.gz", "rb") as f_in:
            with open(csv_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        df = pd.read_csv(csv_file_path)
        df = df.drop(["time"], axis=1)
        df = df.iloc[::4]  # downsample by factor of 4 | 100Hz => 25Hz
        df["annotation"] = df["annotation"].str.split().str[-1].astype(float)
        df["annotation"] = df["annotation"].interpolate()
        np.save(
            capture24_preprocessed + "/" + name,
            df.to_numpy(),
        )
        os.remove(csv_file_path)

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["ieee", "capture24", "ucihar", "dalia"],
        required=True,
        help="You have to choose from [dalia, ucihar, ieee, capture24]",
    )

    parser.add_argument(
        "--datadir", required=True, help="Specifiy the path were the data is located."
    )

    args = parser.parse_args()

    if args.dataset == "capture24":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/Capture24/capture24/"
        create_capture24_npy_files(args.datadir)
    elif args.dataset == "dalia":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
        create_dalia_npy_files(args.datadir)
    elif args.dataset == "ucihar":
        # datadir = (
        #      "C:/Users/cleme/ETH/Master/Thesis/data/UCIHAR/UCI HAR Dataset/UCI HAR Dataset/"
        # )
        ucihar_preprocess(args.datadir)
    elif args.dataset == "ieee":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/IEEEPPG/Training_data/Training_data"
        create_ieee_npz_files(args.datadir)
    else:
        raise NotImplementedError()
