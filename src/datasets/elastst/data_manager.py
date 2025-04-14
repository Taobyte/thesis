import re
import torch
import numpy as np
import pandas as pd
from functools import cached_property

from gluonts.dataset.common import ListDataset

from src.datasets.elastst.datasets.multi_horizon_datasets import MultiHorizonDataset
from src.datasets.elastst.datasets.single_horizon_datasets import SingleHorizonDataset

from src.datasets.elastst.data_utils.time_features import get_lags
from src.datasets.elastst.data_utils.data_utils import (
    split_train_val,
    truncate_test,
    get_rolling_test,
    df_to_mvds,
)
from src.datasets.elastst.data_wrapper import ProbTSBatchData
from src.datasets.elastst.data_utils.data_scaler import (
    StandardScaler,
    TemporalScaler,
    IdentityScaler,
)
from typing import Union
from src.datasets.dalia_dataset import dalia_load_data
from src.datasets.ieee_dataset import ieee_load_data

from omegaconf import DictConfig

datasets = ["dalia", "ieee", "ucihar", "usc", "wildppg", "capture24", "chapman"]


def convert_to_list(s):
    """
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    """
    if type(s).__name__ == "int":
        return [s]
    elif type(s).__name__ == "list":
        return s
    elif type(s).__name__ == "str":
        elements = re.split(r"\D+", s)
        return list(map(int, elements))
    elif type(s).__name__ == "ListConfig":
        return list(s)
    else:
        return None


def ensure_list(input_value, default_value=None):
    """
    Ensures that the input is converted to a list. If the input is None,
    it converts the default value to a list instead.
    """
    result = convert_to_list(input_value)
    if result is None:
        result = convert_to_list(default_value)
    return result


class DataManager:
    def __init__(
        self,
        dataset_cfg: DictConfig,
        datadir: str = "./datasets",
        history_length: int = None,
        context_length: int = None,
        prediction_length: Union[list, int, str] = None,
        train_ctx_len: int = None,
        train_pred_len_list: Union[list, int, str] = None,
        val_ctx_len: int = None,
        val_pred_len_list: Union[list, int, str] = None,
        test_rolling_length: int = 96,
        split_val: bool = True,
        scaler: str = "none",
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
        data_path: str = None,
        freq: str = None,
        multivariate: bool = True,
        continuous_sample: bool = False,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        auto_search: bool = False,
    ):
        """
        DataManager class for handling datasets and preparing data for time-series models.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load. Examples include "etth1", "electricity_ltsf", etc.
        datadir : str, optional, default='./datasets'
            Root directory datadir where datasets are stored.
        history_length : int, optional, default=None
            Length of the historical input window for the model.
            If not specified, it is automatically calculated based on `context_length` and lag features.
        context_length : int, optional, default=None
            Length of the input context for the model.
        prediction_length : Union[list, int, str], optional, default=None
            Length of the prediction horizon for the model. Can be:
            - int: Fixed prediction length.
            - list: Variable prediction lengths for multi-horizon training.
            - str: The string format of multiple prediction length. E.g., '96-192-336-720' represents [96, 192, 336, 720]
        train_ctx_len : int, optional, default=None
            Context length for the training dataset.
            If not specified, defaults to the value of `context_length`.
        train_pred_len_list : Union[list, int, str], optional, default=None
            List of prediction lengths for the training dataset.
            If not specified, defaults to the value of `prediction_length`.
        val_ctx_len : int, optional, default=None
            Context length for the validation dataset.
            If not specified, defaults to the value of `context_length`.
        val_pred_len_list : Union[list, int, str], optional, default=None
            List of prediction lengths for the validation dataset.
            If not specified, defaults to the value of `prediction_length`.
        test_rolling_length : int, optional, default=96
            Gap window size used for rolling predictions in the testing phase.
            - If set to `auto`, it is dynamically determined based on the dataset frequency
            (e.g., 'H' -> 24, 'D' -> 7, 'W' -> 4).
        split_val : bool, optional, default=True
            Whether to split the training dataset into training and validation sets.
        scaler : str, optional, default='none'
            Type of normalization or scaling applied to the dataset. Options include:
            - 'none': No scaling.
            - 'standard': Standard normalization (z-score).
            - 'temporal': Mean-scaling normalization.
        context_length_factor : int, optional, default=1
            Scaling factor for context length, allowing dynamic adjustment of `context_length`.
        timeenc : int, optional, default=1
            Time encoding strategy. Options include:
            - 0: The dimension of time feature is 5, containing `month, day, weekday, hour, minute`
            - 1: Cyclic time features (e.g., sine/cosine of timestamps).
            - 2: Raw Timestamp information.
        var_specific_norm : bool, optional, default=True
            Whether to normalize variables independently. Only applies when `scaler='standard'`.
        data_path : str, optional, default=None
            Specific datadir to the dataset file.
        freq : str, optional, default=None
            Data frequency (e.g., 'H' for hourly, 'D' for daily).
        multivariate : bool, optional, default=True
            Whether the dataset is multivariables.
        continuous_sample : bool, optional, default=False
            Whether to enable continuous sampling for forecasting horizons during training phase.
        train_ratio : float, optional, default=0.7
            Proportion of the dataset used for training. Default is 70% of the data.
        test_ratio : float, optional, default=0.2
            Proportion of the dataset used for testing. Default is 20% of the data.
        auto_search : bool, optional, default=False
            Make past_len=ctx_len+pred_len, enabling post training search.
        """

        self.dataset_cfg = dataset_cfg  # OmegeConf dictionary of the dataset
        self.datadir = datadir
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.train_ctx_len = (
            train_ctx_len if train_ctx_len is not None else context_length
        )
        self.val_ctx_len = val_ctx_len if val_ctx_len is not None else context_length
        self.train_pred_len_list = (
            train_pred_len_list
            if train_pred_len_list is not None
            else prediction_length
        )
        self.val_pred_len_list = (
            val_pred_len_list if val_pred_len_list is not None else prediction_length
        )
        self.test_rolling_length = test_rolling_length
        self.split_val = split_val
        self.scaler_type = scaler
        self.context_length_factor = context_length_factor
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm
        self.data_path = data_path
        self.freq = freq
        self.multivariate = multivariate
        self.continuous_sample = continuous_sample
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.auto_search = auto_search

        self.test_rolling_dict = {"h": 24, "d": 7, "b": 5, "w": 4, "min": 60}
        self.global_mean = None

        # Configure scaler
        self.scaler = self._configure_scaler(self.scaler_type)

        # Process context and prediction lengths
        self._process_context_and_prediction_lengths()
        self.prepare_dataset()
        # Print configuration details
        self._print_configurations()

    def _configure_scaler(self, scaler_type: str):
        """Configure the scaler."""
        if scaler_type == "standard":
            return StandardScaler(var_specific=self.var_specific_norm)
        elif scaler_type == "temporal":
            return TemporalScaler()
        return IdentityScaler()

    def _set_meta_parameters(self, target_dim, freq, prediction_length):
        """Set meta parameters from base dataset."""
        self.target_dim = target_dim
        self.multivariate = self.target_dim > 1
        self.freq = freq
        self.lags_list = get_lags(self.freq)
        self.prediction_length = prediction_length
        self.context_length = (
            self.context_length or self.prediction_length * self.context_length_factor
        )
        self.history_length = self.history_length or (
            self.context_length + max(self.lags_list)
        )

    def _process_context_and_prediction_lengths(self):
        """Convert context and prediction lengths to lists for multi-horizon processing."""

        self.train_ctx_len_list = ensure_list(
            self.train_ctx_len, default_value=self.context_length
        )
        self.val_ctx_len_list = ensure_list(
            self.val_ctx_len, default_value=self.context_length
        )
        self.test_ctx_len_list = ensure_list(self.context_length)
        self.train_pred_len_list = ensure_list(
            self.train_pred_len_list, default_value=self.prediction_length
        )
        self.val_pred_len_list = ensure_list(
            self.val_pred_len_list, default_value=self.prediction_length
        )
        self.test_pred_len_list = ensure_list(self.prediction_length)

        # Validate context length support
        assert len(self.train_ctx_len_list) == 1, (
            "Assign a single context length for training."
        )
        assert len(self.val_ctx_len_list) == 1, (
            "Assign a single context length for validation."
        )
        assert len(self.test_ctx_len_list) == 1, (
            "Assign a single context length for testing."
        )

        self.multi_hor = (
            len(self.train_pred_len_list) > 1
            or len(self.val_pred_len_list) > 1
            or len(self.test_pred_len_list) > 1
        )

    def _set_meta_parameters_from_raw(self, data_size):
        """Set meta parameters directly from raw dataset."""
        self.lags_list = get_lags(self.freq)
        self.prediction_length = (
            ensure_list(self.prediction_length)
            if self.multi_hor
            else self.prediction_length
        )
        self.context_length = (
            ensure_list(self.context_length) if self.multi_hor else self.context_length
        )
        self.history_length = self.history_length or (
            max(self.context_length) + max(self.lags_list)
            if self.multi_hor
            else self.context_length + max(self.lags_list)
        )
        if not self.multivariate:
            self.target_dim = 1
            raise NotImplementedError(
                "Customized univariate datasets are not yet supported."
            )

        # define the test_rolling_length
        if self.test_rolling_length == "auto":
            if self.freq.lower() in self.test_rolling_dict:
                self.test_rolling_length = self.test_rolling_dict[self.freq.lower()]
            else:
                self.test_rolling_length = 24

    def _dalia_prepare_dataset(self):
        dalia_data = dalia_load_data(
            self.datadir,
            [1, 2, 3],
        )
        return dalia_data

    def _ieee_prepare_dataset(self):
        def _create_dataset(series: list):
            start = str(pd.Timestamp("2022-01-01 00:00:00"))
            processed_series = []
            if self.dataset_cfg.datamodule.use_heart_rate:
                for serie in series:
                    processed_series.append(
                        {
                            "target": serie[np.newaxis, :],
                            "start": start,
                        }
                    )

            else:
                for participant in range(len(series)):
                    participant_series = series[participant]
                    for serie in range(len(participant_series)):
                        processed_series.append(
                            {
                                "target": series[participant][serie][np.newaxis, :],
                                "start": start,
                            }
                        )
                        print(series[participant][serie].T.shape)
                        break

            return ListDataset(processed_series, freq="20ms", one_dim_target=False)

        train_ieee_series = ieee_load_data(
            self.dataset_cfg.datamodule.data_dir,
            self.dataset_cfg.datamodule.train_participants,
            self.dataset_cfg.datamodule.use_heart_rate,
        )
        val_ieee_series = ieee_load_data(
            self.dataset_cfg.datamodule.data_dir,
            self.dataset_cfg.datamodule.val_participants,
            self.dataset_cfg.datamodule.use_heart_rate,
        )
        test_ieee_series = ieee_load_data(
            self.dataset_cfg.datamodule.data_dir,
            self.dataset_cfg.datamodule.test_participants,
            self.dataset_cfg.datamodule.use_heart_rate,
        )
        train_dataset = _create_dataset(train_ieee_series)
        val_dataset = _create_dataset(val_ieee_series)
        test_dataset = _create_dataset(test_ieee_series)

        self.target_dim = 1

        return train_dataset, val_dataset, test_dataset

    def prepare_dataset(self):
        """Prepare datasets for training, validation, and testing."""

        if self.dataset_cfg.name == "dalia":
            # self._dalia_prepare_dataset()
            raise NotImplementedError
        elif self.dataset_cfg.name == "ieee":
            group_train_set, group_val_set, group_test_set = (
                self._ieee_prepare_dataset()
            )

        # TODO: MAYBE IMPLEMENT DATA SCALING

        # set lags list parameter
        self.lags_list = [1, 2, 3]  # TODO: calculate lags based on freq

        if self.multi_hor:
            # Handle multi-horizon datasets
            dataset_loader = self._prepare_multi_horizon_datasets(
                group_val_set, group_test_set
            )
        else:
            # Handle single-horizon datasets
            dataset_loader = self._prepare_single_horizon_datasets(
                group_val_set, group_test_set
            )

        self.dataset_loader = dataset_loader

        self.train_iter_dataset = dataset_loader.get_iter_dataset(
            group_train_set,
            mode="train",
        )

        self.time_feat_dim = dataset_loader.time_feat_dim
        self.global_mean = torch.mean(
            torch.tensor(group_train_set[0]["target"]), dim=-1
        )

    def _prepare_multi_horizon_datasets(self, group_val_set, group_test_set):
        """Prepare multi-horizon datasets for validation and testing."""
        self.val_iter_dataset = {}
        self.test_iter_dataset = {}
        dataset_loader = MultiHorizonDataset(
            input_names=ProbTSBatchData.input_names_,
            freq=self.freq,
            train_ctx_range=self.train_ctx_len_list,
            train_pred_range=self.train_pred_len_list,
            val_ctx_range=self.val_ctx_len_list,
            val_pred_range=self.val_pred_len_list,
            test_ctx_range=self.test_ctx_len_list,
            test_pred_range=self.test_pred_len_list,
            multivariate=self.multivariate,
            continuous_sample=self.continuous_sample,
        )

        # Prepare validation datasets
        for pred_len in self.val_pred_len_list:
            self.val_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                group_val_set,
                mode="val",
                pred_len=[pred_len],
            )

        # Prepare testing datasets
        for pred_len in self.test_pred_len_list:
            self.test_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                group_test_set,
                mode="test",
                pred_len=[pred_len],
                auto_search=self.auto_search,
            )

        return dataset_loader

    def _prepare_single_horizon_datasets(self, group_val_set, group_test_set):
        """Prepare single-horizon datasets for training, validation, and testing."""
        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_,
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate,
        )

        # Validation dataset
        self.val_iter_dataset = dataset_loader.get_iter_dataset(
            group_val_set,
            mode="val",
        )

        # Testing dataset
        self.test_iter_dataset = dataset_loader.get_iter_dataset(
            group_test_set,
            mode="test",
            auto_search=self.auto_search,
        )

        return dataset_loader

    def _print_configurations(self):
        """Print dataset and configuration details."""
        print(
            f"Test context length: {self.test_ctx_len_list}, prediction length: {self.test_pred_len_list}"
        )
        print(
            f"Validation context length: {self.val_ctx_len_list}, prediction length: {self.val_pred_len_list}"
        )
        print(
            f"Training context length: {self.train_ctx_len_list}, prediction lengths: {self.train_pred_len_list}"
        )
        print(f"Test rolling length: {self.test_rolling_length}")
        if self.scaler_type == "standard":
            print(f"Variable-specific normalization: {self.var_specific_norm}")
