import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy

from pathlib import Path
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.io import loadmat


# -------------------------------------------------------------------
# WildPPG
# -------------------------------------------------------------------


def load_wildppg_participant(path):
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


def panPeakDetect(detection, fs):
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
                            rr < min(hb_intervals) / tol
                            and rr > max(hb_intervals) * tol
                        ):
                            good_rrs.append(rr)
                            good_rrs_x.append((good_hbeats[-1] + good_hbeats[-2]) / 2)
    return np.array(good_hbeats), np.array(good_rrs), np.array(good_rrs_x)


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


def preprocess_wildppg_mat_file(datadir: str, winsizes: list[int] = [8]) -> None:
    for winsize in winsizes:
        all_hrs = []
        all_imus = []
        all_enmos = []
        all_mads = []
        all_rmses = []
        all_percs = []
        all_jerks = []
        all_rmse_last2 = []
        all_cadences = []
        for pidx, p in enumerate(Path(datadir).iterdir()):
            print(pidx, " load ", p)
            part_data = load_wildppg_participant(p.absolute())
            r_peaks = pan_tompkins_detector(
                part_data["sternum"]["ecg"]["v"], part_data["sternum"]["ecg"]["fs"]
            )
            ecgpks_filt, rrs, rrxs = quotient_filter(r_peaks, outlier_over=5, tol=0.75)

            x = part_data["ankle"]["acc_x"]["v"]
            y = part_data["ankle"]["acc_y"]["v"]
            z = part_data["ankle"]["acc_z"]["v"]
            imu = np.sqrt(x**2 + y**2 + z**2)

            fs = part_data["sternum"]["ecg"]["fs"]

            hrs = []
            imus = []
            enmos = []
            mads = []
            percs = []
            rmses = []
            jerks = []
            rmse_last2s = []
            cadences = []
            for win_s in tqdm(range(0, max(ecgpks_filt), winsize * fs)):
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

                window = imu[win_s : win_s + winsize * fs]
                mean = np.mean(window)
                rmse = np.sqrt(np.mean(np.square(window)))
                enmo = np.mean(np.maximum(window - 1.0, 0.0))
                mad = np.mean(np.abs(window - mean))
                p90 = np.percentile(window, 90)
                jerk = np.sqrt(np.mean(np.diff(window) ** 2)) * fs
                rmse_last2 = np.sqrt(np.mean(window[-2 * fs :] ** 2))

                f, Pxx = scipy.signal.welch(
                    window, fs=fs, nperseg=min(len(window), 512)
                )
                Pxx = Pxx / (np.trapezoid(Pxx, f) + 1e-12)

                band_loco = (f >= 0.5) & (f <= 3.0)
                cadence = (
                    f[band_loco][np.argmax(Pxx[band_loco])] if band_loco.any() else 0.0
                )

                imus.append(mean)
                rmses.append(rmse)
                enmos.append(enmo)
                mads.append(mad)
                percs.append(p90)
                jerks.append(jerk)
                rmse_last2s.append(rmse_last2)
                cadences.append(cadence)

            hrs_col = np.asarray(hrs, dtype=np.float32)[:, None]
            imus_col = np.asarray(imus, dtype=np.float32)[:, None]
            rmse_col = np.asarray(rmses, dtype=np.float32)[:, None]
            enmo_col = np.asarray(enmos, dtype=np.float32)[:, None]
            mad_col = np.asarray(mads, dtype=np.float32)[:, None]
            perc_col = np.asarray(percs, dtype=np.float32)[:, None]
            jerk_col = np.asarray(jerks, dtype=np.float32)[:, None]
            rmse_last2_col = np.asarray(rmse_last2s, dtype=np.float32)[:, None]
            cadence_col = np.asarray(cadences, dtype=np.float32)[:, None]

            all_hrs.append(hrs_col)
            all_imus.append(imus_col)
            all_rmses.append(rmse_col)
            all_enmos.append(enmo_col)
            all_mads.append(mad_col)
            all_percs.append(perc_col)
            all_jerks.append(jerk_col)
            all_rmse_last2.append(rmse_last2_col)
            all_cadences.append(cadence_col)

        data_bpm_values = np.empty((len(all_hrs), 1), dtype=object)
        data_imu_ankle = np.empty((len(all_imus), 1), dtype=object)
        rmse = np.empty((len(all_imus), 1), dtype=object)
        enmo = np.empty((len(all_imus), 1), dtype=object)
        mad = np.empty((len(all_imus), 1), dtype=object)
        perc = np.empty((len(all_imus), 1), dtype=object)
        jerk = np.empty((len(all_imus), 1), dtype=object)
        rmse_last2 = np.empty((len(all_imus), 1), dtype=object)
        cadence = np.empty((len(all_imus), 1), dtype=object)
        for i in range(len(all_hrs)):
            data_bpm_values[i, 0] = all_hrs[i]
            data_imu_ankle[i, 0] = all_imus[i]
            rmse[i, 0] = all_rmses[i]
            enmo[i, 0] = all_enmos[i]
            mad[i, 0] = all_mads[i]
            perc[i, 0] = all_percs[i]
            jerk[i, 0] = all_jerks[i]
            rmse_last2[i, 0] = all_rmse_last2[i]
            cadence[i, 0] = all_cadences[i]
        outdict = {
            "data_bpm_values": data_bpm_values,
            "data_imu_ankle": data_imu_ankle,
            "rmse": rmse,
            "enmo": enmo,
            "mad": mad,
            "perc": perc,
            "jerk": jerk,
            "rmse_last2": rmse_last2,
            "cadence": cadence,
        }

        scipy.io.savemat(f"./data/WildPPG_{winsize}.mat", outdict)


# -------------------------------------------------------------------
# IEEE SPC
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# DALIA
# -------------------------------------------------------------------


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
            imu_var = np.var(windows, axis=1)
            imu_power = np.mean(windows**2, axis=1)
            imu_energy = np.sum(windows**2, axis=1)
            imu_rms = np.sqrt(np.mean(windows**2, axis=1))

            assert heart_rate.shape == acc_norm_heart_rate.shape

            activity = data["activity"]
            np.savez(
                dalia_preprocessed_dir + "/" + (str(path).split("\\")[-2]),
                bvp=bvp,
                heart_rate=heart_rate,
                acc_norm_ppg=acc_norm_ppg,
                acc_norm_heart_rate=acc_norm_heart_rate,
                imu_var=imu_var,
                imu_power=imu_power,
                imu_energy=imu_energy,
                imu_rms=imu_rms,
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


def main():
    parser = argparse.ArgumentParser()

    def list_of_ints(arg: str) -> list[int]:
        return [int(i) for i in arg.split(",")]

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
        "--winsizes",
        type=list_of_ints,
        required=False,
        default=[8],
        help="Sliding window size for WildPPG",
    )

    args = parser.parse_args()

    if args.dataset == "dalia":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
        create_dalia_npy_files(args.datadir)
    elif args.dataset == "ieee":
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/IEEEPPG/Training_data/Training_data"
        create_ieee_npz_files(args.datadir)
    elif args.dataset == "wildppg":
        # C:/Users/cleme/ETH/Master/Thesis/data/WildPPG/data
        preprocess_wildppg_mat_file(args.datadir, args.winsizes)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
