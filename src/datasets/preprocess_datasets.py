import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy
import pycatch22

from pathlib import Path
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


def filter_butter(
    x: NDArray[np.float32], fs: int, f1: float = 0.5, f2: float = 5.0, order: int = 4
) -> NDArray[np.float32]:
    Wn = [f1, f2]
    sos = butter(order, Wn, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    return filtered


def process_acc_signal(
    signal: NDArray[np.float32],
    window_size: int = 8,
    stride_size: int = 2,
    fs: int = 32,
    filter: bool = True,
    f2: float = 5.0,
    feature_name: str = "mean",
) -> NDArray[np.float32]:
    window = window_size * fs
    stride = stride_size * fs
    filtered = (
        np.column_stack([filter_butter(signal[:, i], fs=fs, f2=f2) for i in range(3)])
        if filter
        else signal
    )
    acc_norm = np.linalg.norm(filtered, axis=1)
    windows = sliding_window_view(acc_norm, window_shape=window)[::stride]
    if feature_name == "mean":
        res = np.mean(windows, axis=1, keepdims=True)
    elif feature_name == "std":
        res = np.std(windows, axis=1, keepdims=True)
    elif feature_name == "rms":
        res = np.sqrt(np.mean(windows**2, axis=1, keepdims=True))
    elif feature_name == "jerk":
        dx = np.diff(windows, axis=1, prepend=windows[:, :1])
        res = np.sqrt((dx**2).mean(axis=1, keepdims=True)) * fs
    elif feature_name == "last2s_rms":
        res = np.sqrt((windows[:, -2 * fs :] ** 2).mean(axis=1, keepdims=True))
    elif feature_name == "centroid":
        _, W = windows.shape
        T = W / fs
        t = (np.arange(W, dtype=float) / fs)[None, :]
        e = windows**2
        E = e.sum(axis=1) + 1e-12
        t_centroid = (e * t).sum(axis=1) / E
        res = t_centroid / T
        res = res[:, None]
    elif feature_name == "catch22":
        catch22_features: list[NDArray[np.float32]] = []
        for w in tqdm(windows):
            w_f = pycatch22.catch22_all(w, catch24=True)
            catch22_features.append(np.array(w_f["values"]))

        res = np.vstack(catch22_features)  # (W,24)

    else:
        raise NotImplementedError(f"{feature_name} not implemented")

    return res


def get_all_features(
    signal: NDArray[np.float32],
    window_size: int = 8,
    stride_size: int = 2,
    fs: int = 32,
    filter: bool = True,
    f2: float = 5.0,
    prefix: str = "",
    features: list[str] = ["mean", "std"],
) -> dict[str, NDArray[np.float32]]:
    traces: dict[str, NDArray[np.float32]] = {}
    for feature in features:
        processed_trace = process_acc_signal(
            signal,
            window_size=window_size,
            stride_size=stride_size,
            fs=fs,
            filter=filter,
            f2=f2,
            feature_name=feature,
        )
        traces[prefix + feature] = processed_trace

    return traces


# -------------------------------------------------------------------
# WildPPG
# -------------------------------------------------------------------


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(
        order, [low, high], analog=False, btype="band", output="sos"
    )
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.sosfiltfilt(sos, data)
    return y


def load_wildppg_participant(path: Path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries
    """
    loaded_data = scipy.io.loadmat(path)
    loaded_data["id"] = loaded_data["id"][0]
    if len(loaded_data["notes"]) == 0:
        loaded_data["notes"] = ""
    else:
        loaded_data["notes"] = loaded_data["notes"][0]

    for bodyloc in ["sternum", "head", "wrist", "ankle"]:
        bodyloc_data = dict()  # data structure to feed cleaned data into
        sensors = loaded_data[bodyloc][0].dtype.names
        for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
            bodyloc_data[sensor_name] = dict()
            field_names = sensor_data[0][0].dtype.names
            for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                bodyloc_data[sensor_name][sensor_field] = field_data[0]
                if sensor_field == "fs":
                    bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][
                        sensor_field
                    ][0]
        loaded_data[bodyloc] = bodyloc_data
    return loaded_data


def panPeakDetect(detection, fs: int):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.

    Original implementation by Luis Howell luisbhowell@gmail.com, Bernd Porr, bernd.porr@glasgow.ac.uk, DOI: 10.5281/zenodo.3353396
    """
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    indexes = []

    missed_peaks = []
    peaks = scipy.signal.find_peaks(detection, distance=min_distance)[0]

    thres_weight = 0.125

    for index, peak in enumerate(peaks):
        if peak > 4 * fs and threshold_I1 > max(
            detection[peak - 4 * fs : peak]
        ):  # reset thresholds if we do not see any peaks anymore
            SPKI_n = max(detection[peak - 4 * fs : peak])
            NPKI = min(
                NPKI * SPKI_n / SPKI, np.percentile(detection[peak - 4 * fs : peak], 80)
            )
            SPKI = SPKI_n
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1

        if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:
            signal_peaks.append(peak)
            indexes.append(index)
            SPKI = (
                thres_weight * detection[signal_peaks[-1]] + (1 - thres_weight) * SPKI
            )
            if RR_missed != 0:
                if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                    missed_section_peaks = peaks[indexes[-2] + 1 : indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if (
                            missed_peak - signal_peaks[-2] > min_distance
                            and signal_peaks[-1] - missed_peak > min_distance
                            and detection[missed_peak] > threshold_I2
                        ):
                            missed_section_peaks2.append(missed_peak)

                    if len(missed_section_peaks2) > 0:
                        signal_missed = [detection[i] for i in missed_section_peaks2]
                        index_max = np.argmax(signal_missed)
                        missed_peak = missed_section_peaks2[index_max]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak
            if len(signal_peaks) > 100 and thres_weight > 0.1:
                thres_weight = 0.0125

        else:
            noise_peaks.append(peak)
            NPKI = thres_weight * detection[noise_peaks[-1]] + (1 - thres_weight) * NPKI

        threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold_I2 = 0.5 * threshold_I1

        if len(signal_peaks) > 8:
            RR = np.diff(signal_peaks[-9:])
            RR_ave = int(np.mean(RR))
            RR_missed = int(1.66 * RR_ave)

    signal_peaks.pop(0)

    return signal_peaks


def pan_tompkins_detector(unfiltered_ecg, sr):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.

    Original implementation by Luis Howell luisbhowell@gmail.com, Bernd Porr, bernd.porr@glasgow.ac.uk, DOI: 10.5281/zenodo.3353396
    """
    maxQRSduration = 0.150  # sec
    filtered_ecg = butter_bandpass_filter(unfiltered_ecg, 5, 15, sr, order=1)

    diff = np.diff(filtered_ecg)
    squared = diff * diff

    mwa = scipy.ndimage.uniform_filter1d(squared, size=int(maxQRSduration * sr))
    # cap mwa during motion artefacts to make sure it does not screw the thresholds
    maxvals = (
        scipy.ndimage.maximum_filter1d(filtered_ecg, size=int(maxQRSduration * sr))[:-1]
        / 400
    )
    mwa = np.asarray([v if v < maxval else maxval for maxval, v in zip(maxvals, mwa)])

    mwa[: int(maxQRSduration * sr * 2)] = 0

    searchr = int(maxQRSduration * sr)
    peakfind = butter_bandpass_filter(unfiltered_ecg, 7.5, 20, sr, order=1)

    mwa_peaks = panPeakDetect(mwa, sr)
    r_peaks2 = []
    for rp in mwa_peaks:
        r_peaks2.append(
            rp - searchr + np.argmax(peakfind[rp - searchr : rp + searchr + 1])
        )
    r_peaks3 = []
    for rp in r_peaks2:
        r_peaks3.append(
            rp - 2 + np.argmax(unfiltered_ecg[rp - 2 : rp + 3])
        )  # adjust by at most 2 samples to hit raw data max
    return np.asarray(r_peaks3)


def quotient_filter(hbpeaks, outlier_over=5, sampling_rate=128, tol=0.8):
    """
    Function that applies a quotient filter similar to what is described in
    "Piskorki, J., Guzik, P. (2005), Filtering Poincare plots"
    it preserves peaks that are part of a sequence of [outlier_over] peaks with
    a tolerance of [tol]"""
    good_hbeats = []
    good_rrs = []
    good_rrs_x = []
    for i, peak in enumerate(hbpeaks[: -(outlier_over - 1)]):
        hb_intervals = [
            hbpeaks[j] - hbpeaks[j - 1] for j in range(i + 1, i + outlier_over)
        ]
        hr = 60 / ((sum(hb_intervals)) / ((outlier_over - 1) * sampling_rate))
        if (
            min(hb_intervals) > max(hb_intervals) * tol and hr > 35 and hr < 185
        ):  # -> good data
            for p in hbpeaks[i : i + outlier_over]:
                if len(good_hbeats) == 0 or p > good_hbeats[-1]:
                    good_hbeats.append(p)
                    if len(good_hbeats) > 1:
                        rr = good_hbeats[-1] - good_hbeats[-2]
                        if (
                            max(hb_intervals) * tol < rr
                            and rr < min(hb_intervals) / tol
                        ):
                            good_rrs.append(rr)
                            good_rrs_x.append((good_hbeats[-1] + good_hbeats[-2]) / 2)
    return np.array(good_hbeats), np.array(good_rrs), np.array(good_rrs_x)


def preprocess_wildppg_mat_file(
    datadir: str, imu_features: list[str] = ["rms"]
) -> None:
    winsize = 8  # 8s window size
    stride = 2  # 2s stride
    all_hrs: list[NDArray[np.float32]] = []
    all_imus: list[dict[str, NDArray[np.float32]]] = []
    for pidx, p in enumerate(Path(datadir).iterdir()):
        print(pidx, " load ", p)
        part_data = load_wildppg_participant(p.absolute())

        r_peaks = pan_tompkins_detector(
            part_data["sternum"]["ecg"]["v"], part_data["sternum"]["ecg"]["fs"]
        )
        ecgpks_filt, rrs, rrxs = quotient_filter(r_peaks, outlier_over=5, tol=0.75)

        def get_imu_by_location(location: str):
            assert location in ["wrist", "sternum", "ankle", "head"]
            x = part_data[location]["acc_x"]["v"]
            y = part_data[location]["acc_y"]["v"]
            z = part_data[location]["acc_z"]["v"]
            imu = np.stack([x, y, z], axis=-1)
            return imu

        ankle_imu = get_imu_by_location("ankle")

        fs = part_data["sternum"]["ecg"]["fs"]

        ankle_dict = get_all_features(
            ankle_imu, fs=128, prefix="ankle_", features=imu_features
        )

        hrs: list[NDArray[np.float32]] = []
        for win_s in tqdm(range(0, max(ecgpks_filt), stride * fs)):
            rr_in_win = rrs[
                np.logical_and(
                    rrxs > win_s,
                    rrxs < win_s + winsize * fs,
                )
            ]
            if len(rr_in_win) > 1:  # at least 2
                hrs.append(60 * len(rr_in_win) / (np.sum(rr_in_win) / fs))
            else:
                hrs.append(0)  # invalid / noisy ecg

        ankle_length = len(ankle_dict["ankle_rms"])
        hr_array = np.array(hrs)
        hr_length = len(hrs)
        min_length = min(hr_length, ankle_length)
        hr_array = hr_array[:min_length, None]
        for k, v in ankle_dict.items():
            ankle_dict[k] = v[:min_length]

        for k, v in ankle_dict.items():
            assert len(v) == len(hrs)

        all_hrs.append(hr_array)
        all_imus.append(ankle_dict)

    data_bpm_values = np.empty((len(all_hrs), 1), dtype=object)
    for i in range(len(all_hrs)):
        data_bpm_values[i, 0] = all_hrs[i]

    hr_arr = np.array(all_hrs, dtype=object)  # shape: (N_participants,)
    imus_arr = np.array(all_imus, dtype=object)  # shape: (N_participants,)

    np.savez(
        f"./data/WildPPG_{winsize}.npz", hr=hr_arr, imus=imus_arr, allow_pickle=True
    )


# -------------------------------------------------------------------
# IEEE SPC
# -------------------------------------------------------------------


def create_ieee_npz_files(datadir: str, features: list[str] = ["mean"]):
    ieee_preprocessed = os.path.join(datadir, "ieee_filtered")
    os.makedirs(ieee_preprocessed, exist_ok=True)
    signal_files = Path(datadir).glob("*[12].mat")
    bpm_files = Path(datadir).glob("*_BPMtrace.mat")

    print("Start processing IEEE files...")
    for i, (signal_file, bpm_file) in enumerate(zip(signal_files, bpm_files)):
        signals = loadmat(signal_file)["sig"]
        bpm = loadmat(bpm_file)["BPM0"]  # (H, 1)

        acc_xyz = np.vstack([signals[3], signals[4], signals[5]]).T

        imu_features = get_all_features(
            acc_xyz, fs=125, features=features, prefix="wrist_"
        )

        for _, v in imu_features.items():
            assert len(v) == len(bpm)

        np.savez(ieee_preprocessed + "/" + f"IEEE_{i}", bpms=bpm, **imu_features)

    print("End processing IEEE files.")


# -------------------------------------------------------------------
# DALIA
# -------------------------------------------------------------------


def create_dalia_npy_files(datadir: str, features: list[str] = ["mean"]):
    dalia_preprocessed_dir = os.path.join(datadir, "dalia_filtered_preprocessed")
    os.makedirs(dalia_preprocessed_dir, exist_ok=True)

    print("Start processing dalia files.")
    for path in tqdm(Path(datadir).glob("**/S*.pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

            hr = data["label"]
            activity = data["activity"]
            wrist_acc = data["signal"]["wrist"]["ACC"]
            chest_acc = data["signal"]["chest"]["ACC"]

            wrist_imu_dict = get_all_features(
                wrist_acc, fs=32, prefix="wrist_", features=features
            )  # Empatica E4 has sampling rate 32Hz
            chest_imu_dict = get_all_features(
                chest_acc, fs=700, prefix="chest_", features=features
            )  # RespiBAN has sampling rate 700Hz
            combined = {**wrist_imu_dict, **chest_imu_dict}

            np.savez(
                dalia_preprocessed_dir + "/" + (str(path).split("\\")[-2]),
                hr=hr,
                activity=activity,
                **combined,
            )

    print("Finished processing dalia files.")


def main():
    parser = argparse.ArgumentParser()

    def list_of_strings(arg: str) -> list[str]:
        return arg.split(",")

    parser.add_argument(
        "--dataset",
        choices=["ieee", "dalia", "wildppg"],
        required=True,
        help="You have to choose from [dalia, wildppg, ieee]",
    )

    parser.add_argument(
        "--datadir", required=True, help="Specifiy the path were the data is located."
    )

    parser.add_argument(
        "--features",
        type=list_of_strings,
        required=False,
        default=["rms", "last2s_rms", "centroid", "jerk"],
        help="IMU Features to create from the raw IMU signal.(See process_acc_signal() for the different features to choose from)",
    )

    args = parser.parse_args()

    if args.dataset == "dalia":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
        create_dalia_npy_files(args.datadir, features=args.features)
    elif args.dataset == "ieee":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/IEEEPPG/Training_data/Training_data"
        create_ieee_npz_files(args.datadir, features=args.features)
    elif args.dataset == "wildppg":
        # C:/Users/cleme/ETH/Master/Thesis/data/WildPPG/data
        preprocess_wildppg_mat_file(args.datadir, imu_features=args.features)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
