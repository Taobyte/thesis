import glob
import pandas as pd
import pickle
import argparse
import os
import gzip
import shutil
import wfdb

from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm


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
        name = zip_file.stem
        name = zip_file.stem.split(".")[0]
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

    print("Start processing static participant features.")

    participant_paths = glob.glob(
        os.path.join(datadir, "**", "*_quest.csv"), recursive=True
    )
    series = []
    for participant in participant_paths:
        row = pd.read_csv(participant, header=None).T
        row.columns = [el.split(" ")[1] for el in row.iloc[0]]
        row = row.drop(row.index[0])
        series.append(row)

    df = pd.concat(series, ignore_index=True)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].str.replace("S", "", regex=True).astype(int)
    df = df.sort_values("SUBJECT_ID").reset_index(drop=True)
    # now we normalize the continuous values and create one-hot encodings
    df[["AGE", "HEIGHT", "WEIGHT"]] = (
        df[["AGE", "HEIGHT", "WEIGHT"]].astype(float)
        - df[["AGE", "HEIGHT", "WEIGHT"]].astype(float).mean()
    ) / (df[["AGE", "HEIGHT", "WEIGHT"]].astype(float).std() + 1e-8)
    df["GENDER"] = df["GENDER"].str.strip().apply(lambda x: 0 if x == "m" else 1)
    one_hot_skin = pd.get_dummies(df["SKIN"], prefix="skin") * 1
    df = pd.concat([df, one_hot_skin], axis=1)
    one_hot_sport = pd.get_dummies(df["SPORT"], prefix="sport") * 1
    df = pd.concat([df, one_hot_sport], axis=1)
    df = df.drop(["SKIN", "SPORT"], axis=1)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].str.strip()
    df.to_csv(datadir + "/static_participant_features.csv")

    print("Finished processing static participant features.")


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


def mhc6mwt_preprocess(datadir: str) -> None:
    walk_hr_df = pd.read_parquet(datadir + "/hr_walk.parquet")
    # rest_hr_df = pd.read_csv(datadir + "/hr_rest.parquet") leave out resting hr for now
    pedometer_df = pd.read_parquet(datadir + "/pedometer.parquet")
    dictionaries = {}

    for id in tqdm(
        walk_hr_df["recordId"].unique(), total=len(walk_hr_df["recordId"].unique())
    ):
        hr_test = walk_hr_df[walk_hr_df["recordId"] == id][["value"]]
        acc_test = pedometer_df[pedometer_df["recordId"] == id]
        acc_test = acc_test.sort_values("endDate")

        # some rows contained endDates < startDate
        acc_test = acc_test[acc_test["startDate"] <= acc_test["endDate"]]

        acc_test = acc_test[
            ["startDate", "endDate", "distance", "numberOfSteps"]
        ]  # distance - > avg speed
        acc_test["seconds"] = (
            (acc_test["endDate"] - acc_test["startDate"]).dt.total_seconds().astype(int)
        )

        acc_test["delta_dist"] = acc_test["distance"].diff()
        acc_test["delta_steps"] = acc_test["numberOfSteps"].diff()
        acc_test["delta_sec"] = acc_test["seconds"].diff()

        acc_test.loc[acc_test.index[0], "delta_sec"] = acc_test.loc[
            acc_test.index[0], "seconds"
        ]
        acc_test.loc[acc_test.index[0], "delta_dist"] = acc_test.loc[
            acc_test.index[0], "distance"
        ]
        acc_test.loc[acc_test.index[0], "delta_steps"] = acc_test.loc[
            acc_test.index[0], "numberOfSteps"
        ]

        # sometimes delta_sec is 0, then we set avg_dist and avg_steps to 0
        acc_test["avg_steps"] = np.where(
            acc_test["delta_sec"] == 0,
            0,
            acc_test["delta_steps"] / acc_test["delta_sec"],
        )
        acc_test["avg_dist"] = np.where(
            acc_test["delta_sec"] == 0,
            0,
            acc_test["delta_dist"] / acc_test["delta_sec"],
        )

        assert (acc_test["delta_sec"] >= 0).all(), (
            f"{np.where(acc_test['delta_sec'].values < 0)}"
        )
        df_repeated = acc_test.loc[
            acc_test.index.repeat(acc_test["delta_sec"])
        ].reset_index(drop=True)
        df_repeated = df_repeated[["avg_steps", "avg_dist"]]
        min_length = min(len(hr_test), len(df_repeated))
        df_concat = pd.concat(
            (hr_test.iloc[:min_length, :], df_repeated.iloc[:min_length, :]), axis=1
        )

        df_concat = df_concat[["value", "avg_dist"]]
        dictionaries[id] = df_concat.values

    dataset = np.concatenate([v[:, 1:] for _, v in dictionaries.items()], axis=0)

    scaler = StandardScaler()
    scaler.fit(dataset)
    mean = scaler.mean_
    std = scaler.scale_

    for k, v in dictionaries.items():
        dictionaries[k] = np.concatenate((v[:, :1], scaler.transform(v[:, 1:])), axis=1)

    dictionaries["z_norm"] = (mean, std)

    with open(datadir + "/mhc6mwt.pkl", "wb") as f:
        pickle.dump(dictionaries, f)


def ptbxl_preprocess(datadir: str) -> None:
    print("Start processing PTB-XL files.")
    # takes approx 21min to load & preprocess all waveforms
    path = Path(datadir)
    ts_dict = {}
    for p in tqdm(path.glob("**/*.dat")):
        record_id = str(p)[-12:-7]
        record = wfdb.rdrecord(str(p)[:-4])
        arr = record.p_signal.astype(np.float32)
        assert arr.shape == (1000, 12)
        ts_dict[record_id] = arr

    np.savez_compressed(path / "ptbxl_preprocessed.npz", **ts_dict)

    print("Finished processing PTB-XL files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["ieee", "capture24", "ucihar", "dalia", "ptbxl", "mhc6mwt"],
        required=True,
        help="You have to choose from [dalia, ucihar, ieee, capture24, ptbxl, mhc6mwt]",
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
    elif args.dataset == "ptbxl":
        ptbxl_preprocess(
            args.datadir
        )  # "C:/Users/cleme/ETH/Master/Thesis/data/PTB/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/"
    elif args.dataset == "mhc6mwt":
        # datadir = "C:\Users\cleme\ETH\Master\Thesis\data\mhc_6mwt_dataset"
        mhc6mwt_preprocess(args.datadir)
    else:
        raise NotImplementedError()
