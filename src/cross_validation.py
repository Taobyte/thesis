import glob
import pandas as pd
import numpy as np
import argparse
import os
import re

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from scipy.io import loadmat


def get_dalia_folds(datadir: str, test_size: int = 5):
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
    df["participant_id"] = df["SUBJECT_ID"].apply(lambda s: int(re.sub("S", "", s)))

    # Bin age & gender into strata
    df["age_bin"] = pd.qcut(df["AGE"].astype(int), q=2, labels=False)
    df["gender_bin"] = df["GENDER"].str.strip().map({"m": 0, "f": 1})
    df["strata"] = df["age_bin"].astype(str) + "_" + df["gender_bin"].astype(str)

    # Deduplicate by participant
    df = df.drop_duplicates("participant_id")

    # Split into test and train_val using stratified sampling
    X_all = df["participant_id"]
    y_all = df["strata"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=42
    )

    test_ids = sorted(X_test.tolist())
    train_val_ids = X_train_val.tolist()
    y_train_val = y_train_val.reset_index(drop=True)

    # Create 3 stratified folds from the remaining train_val participants
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_val_ids, y_train_val)
    ):
        train_ids = [train_val_ids[i] for i in train_idx]
        val_ids = [train_val_ids[i] for i in val_idx]

        folds.append(
            {
                "fold": fold_idx,
                "train_participants": sorted(train_ids),
                "val_participants": sorted(val_ids),
                "test_participants": test_ids,
            }
        )

    # Print summary
    for fold in folds:
        print(f"\nFOLD {fold['fold']}")
        print(f"train_participants: {fold['train_participants']}")
        print(f"val_participants:   {fold['val_participants']}")
        print(f"test_participants:  {fold['test_participants']}")


def get_folds(datadir: str, n_participants: int, train_size: int, test_size: int):
    all_participants = list(range(n_participants))
    np.random.seed(115)
    np.random.shuffle(all_participants)

    test = all_participants[:test_size]
    rest = all_participants[test_size:]

    folds = []
    n_folds = 4

    val_chunks = np.array_split(rest, n_folds)

    for i in range(n_folds - 1):
        val = [int(p) for p in list(val_chunks[i])]
        train = [int(p) for j, chunk in enumerate(val_chunks) if j != i for p in chunk]

        if len(train) > train_size:
            train = train[:train_size]

        folds.append(
            {
                "fold": i,
                "train_participants": sorted(train),
                "val_participants": sorted(val),
                "test_participants": sorted(test),
            }
        )

    for fold in folds:
        print(f"\nFOLD {fold['fold']}")
        print(f"train_participants: {fold['train_participants']}")
        print(f"val_participants:   {fold['val_participants']}")
        print(f"test_participants:  {fold['test_participants']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["ieee", "dalia", "wildppg"],
        required=True,
        help="You have to choose from [dalia, ucihar, ieee, capture24, ptbxl, mhc6mwt, wildppg]",
    )

    parser.add_argument(
        "--datadir", required=True, help="Specify the path were the data is located."
    )

    args = parser.parse_args()

    if args.dataset == "dalia":
        get_dalia_folds(args.datadir)
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy"
    elif args.dataset == "wildppg":
        get_folds(args.datadir, 16, 9, 5)
    elif args.dataset == "ieee":
        get_folds(args.datadir, 12, 6, 4)
        # datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/IEEEPPG/Training_data/Training_data"
    else:
        raise NotImplementedError()
