import itertools
import operator
import functools
import lightning as L
import torch
import numpy as np
import numbers
import random

from torch import Tensor
from jax import grad, vmap
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass

from functools import partial
from dataclasses import is_dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

from src.models.utils import BaseLightningModule

cluster = True
STEP_MULTIPLIER = 1.2

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

loaded = {}


def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.

    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped / base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5 / base**prec

    return sign * val


@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """

    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = " ,"
    bit_sep: str = " "
    plus_sign: str = ""
    minus_sign: str = " -"
    half_bin_correction: bool = True
    decimal_point: str = ""
    missing_str: str = " Nan"


def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), (
        f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"
    )

    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ""
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign

    vnum2repr = partial(
        vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val
    )
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    ismissing = np.isnan(arr)

    def tokenize(arr):
        return "".join([settings.bit_sep + str(b) for b in arr])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0] :]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate(
                    [digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]]
                )
        digits = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += (
        settings.time_sep
    )  # otherwise there is ambiguity in number of digits in the last time step
    return bit_str


def deserialize_str(
    bit_str, settings: SerializerSettings, ignore_last=False, steps=None
):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(
        vec_repr2num,
        base=settings.base,
        prec=settings.prec,
        half_bin_correction=settings.half_bin_correction,
    )
    max_bit_pos = int(np.ceil(np.log(settings.max_val) / np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, (
                    f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
                )
            bit_str = (
                bit_str[len(settings.plus_sign) :]
                if sign == 1
                else bit_str[len(settings.minus_sign) :]
            )
            if settings.bit_sep == "":
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(bits) == max_bit_pos + settings.prec, (
                    f"fixed length bit_str must have {max_bit_pos + settings.prec} bits, but has {len(bits)}: '{bit_str}'"
                )
            digits = []
            for b in bits:
                if b == settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            # digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(
            f"Error deserializing {settings.time_sep.join(bit_strs[i - 2 : i + 5])}{settings.time_sep}\n\t{e}"
        )
        print(f"Got {orig_bitstring}")
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0] * (max_len - len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None


def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    if cluster:
        return "/cluster/project/holz/ckeusch/llama_weights/"

    return "C:/Users/cleme/ETH/Master/Thesis/llama_weights/"


def get_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1
    assert model_size in ["7b", "13b", "70b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        llama2_model_string(model_size, chat),
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_and_tokenizer(model_name, cache_model=False):
    if model_name in loaded:
        return loaded[model_name]
    name_parts = model_name.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]

    tokenizer = get_tokenizer(model_name)

    model = LlamaForCausalLM.from_pretrained(
        llama2_model_string(model_size, chat),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    if cache_model:
        loaded[model_name] = model, tokenizer
    return model, tokenizer


def llama_tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)


def llama_nll_fn(
    model,
    input_arr,
    target_arr,
    settings: SerializerSettings,
    transform,
    count_seps=True,
    temp=1,
    cache_model=True,
):
    """Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
        cache_model: whether to cache the model and tokenizer for faster repeated calls
    Returns: NLL/D
    """
    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out["logits"][:, :, bad_tokens] = -100

    input_ids = batch["input_ids"][0][1:]
    logprobs = torch.nn.functional.log_softmax(out["logits"], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    input_len = len(
        tokenizer(
            [input_str],
            return_tensors="pt",
        )["input_ids"][0]
    )
    input_len = input_len - 2  # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum() / len(target_arr)

    # print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll - avg_logdet_dydx


def llama_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9,
    top_p=0.9,
    cache_model=True,
):
    avg_tokens_per_step = len(llama_tokenize_fn(input_str, model)["input_ids"]) / len(
        input_str.split(settings.time_sep)
    )
    max_tokens = int(avg_tokens_per_step * steps)

    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [input_str],
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch["input_ids"].shape[1]

        good_tokens_str = list("0123456789" + settings.time_sep)
        good_tokens = [
            tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str
        ]
        # good_tokens += [tokenizer.eos_token_id]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return gen_strs


# Required: Text completion function for each model
# -----------------------------------------------
# Each model is mapped to a function that samples text completions.
# The completion function should follow this signature:
#
# Args:
#   - input_str (str): String representation of the input time series.
#   - steps (int): Number of steps to predict.
#   - settings (SerializerSettings): Serialization settings.
#   - num_samples (int): Number of completions to sample.
#   - temp (float): Temperature parameter for model's output randomness.
#
# Returns:
#   - list: Sampled completion strings from the model.
completion_fns = {
    "llama-7b": partial(llama_completion_fn, model="7b"),
    "llama-13b": partial(llama_completion_fn, model="13b"),
    "llama-70b": partial(llama_completion_fn, model="70b"),
    "llama-7b-chat": partial(llama_completion_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_completion_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_completion_fn, model="70b-chat"),
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
# Each model is mapped to a function that computes the continuous Negative Log-Likelihood
# per Dimension (NLL/D). This is used for computing likelihoods only and not needed for sampling.
#
# The NLL function should follow this signature:
#
# Args:
#   - input_arr (np.ndarray): Input time series (history) after data transformation.
#   - target_arr (np.ndarray): Ground truth series (future) after data transformation.
#   - settings (SerializerSettings): Serialization settings.
#   - transform (callable): Data transformation function (e.g., scaling) for determining the Jacobian factor.
#   - count_seps (bool): If True, count time step separators in NLL computation, required if allowing variable number of digits.
#   - temp (float): Temperature parameter for sampling.
#
# Returns:
#   - float: Computed NLL per dimension for p(target_arr | input_arr).
nll_fns = {
    "llama-7b": partial(llama_completion_fn, model="7b"),
    "llama-7b": partial(llama_nll_fn, model="7b"),
    "llama-13b": partial(llama_nll_fn, model="13b"),
    "llama-70b": partial(llama_nll_fn, model="70b"),
    "llama-7b-chat": partial(llama_nll_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_nll_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_nll_fn, model="70b-chat"),
}

# Optional: Tokenization function for each model, only needed if you want automatic input truncation.
# The tokenization function should follow this signature:
#
# Args:
#   - str (str): A string to tokenize.
# Returns:
#   - token_ids (list): A list of token ids.
tokenization_fns = {
    "llama-7b": partial(llama_tokenize_fn, model="7b"),
    "llama-13b": partial(llama_tokenize_fn, model="13b"),
    "llama-70b": partial(llama_tokenize_fn, model="70b"),
    "llama-7b-chat": partial(llama_tokenize_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_tokenize_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_tokenize_fn, model="70b-chat"),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    "llama-7b": 4096,
    "llama-13b": 4096,
    "llama-70b": 4096,
    "llama-7b-chat": 4096,
    "llama-13b-chat": 4096,
    "llama-70b-chat": 4096,
}


class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")

    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


class NoGetItLambdaDict(dict):
    """Regular dict, but refuses to __getitem__ pretending
    the element is not there and throws a KeyError
    if the value is a non string iterable or a lambda"""

    def __init__(self, d={}):
        super().__init__()
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = NoGetItLambdaDict(v)
            else:
                self[k] = v

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value) and value.__name__ == "<lambda>":
            raise LookupError(
                "You shouldn't try to retrieve lambda {} from this dict".format(value)
            )
        if isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, dict, tuple)
        ):
            raise LookupError(
                "You shouldn't try to retrieve iterable {} from this dict".format(value)
            )
        return value


def sample_config(config_spec):
    """Generates configs from the config spec.
    It will apply lambdas that depend on the config and sample from any
    iterables, make sure that no elements in the generated config are meant to
    be iterable or lambdas, strings are allowed."""
    cfg_all = config_spec
    more_work = True
    i = 0
    while more_work:
        cfg_all, more_work = _sample_config(cfg_all, NoGetItLambdaDict(cfg_all))
        i += 1
        if i > 10:
            raise RecursionError(
                "config dependency unresolvable with {}".format(cfg_all)
            )
    out = defaultdict(dict)
    out.update(cfg_all)
    return out


def _sample_config(config_spec, cfg_all):
    cfg = {}
    more_work = False
    for k, v in config_spec.items():
        if isinstance(v, dict):
            new_dict, extra_work = _sample_config(v, cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v, Iterable) and not isinstance(v, (str, bytes, dict, tuple)):
            cfg[k] = random.choice(v)
        elif callable(v) and v.__name__ == "<lambda>":
            try:
                cfg[k] = v(cfg_all)
            except (KeyError, LookupError, Exception):
                cfg[k] = v  # is used isntead of the variable it returns
                more_work = True
        else:
            cfg[k] = v
    return cfg, more_work


def flatten(d, parent_key="", sep="/"):
    """An invertible dictionary flattening operation that does not clobber objs"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and v:  # non-empty dict
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(d, sep="/"):
    """Take a dictionary with keys {'k1/k2/k3':v} to {'k1':{'k2':{'k3':v}}}
    as outputted by flatten"""
    out_dict = {}
    for k, v in d.items():
        if isinstance(k, str):
            keys = k.split(sep)
            dict_to_modify = out_dict
            for partial_key in keys[:-1]:
                try:
                    dict_to_modify = dict_to_modify[partial_key]
                except KeyError:
                    dict_to_modify[partial_key] = {}
                    dict_to_modify = dict_to_modify[partial_key]
                # Base level reached
            if keys[-1] in dict_to_modify:
                dict_to_modify[keys[-1]].update(v)
            else:
                dict_to_modify[keys[-1]] = v
        else:
            out_dict[k] = v
    return out_dict


class grid_iter(object):
    """Defines a length which corresponds to one full pass through the grid
    defined by grid variables in config_spec, but the iterator will continue iterating
    past that by repeating over the grid variables"""

    def __init__(self, config_spec, num_elements=-1, shuffle=True):
        self.cfg_flat = flatten(config_spec)
        is_grid_iterable = lambda v: (
            isinstance(v, Iterable) and not isinstance(v, (str, bytes, dict, tuple))
        )
        iterables = sorted(
            {k: v for k, v in self.cfg_flat.items() if is_grid_iterable(v)}.items()
        )
        if iterables:
            self.iter_keys, self.iter_vals = zip(*iterables)
        else:
            self.iter_keys, self.iter_vals = [], [[]]
        self.vals = list(itertools.product(*self.iter_vals))
        if shuffle:
            with FixedNumpySeed(0):
                random.shuffle(self.vals)
        self.num_elements = (
            num_elements if num_elements >= 0 else (-1 * num_elements) * len(self)
        )

    def __iter__(self):
        self.i = 0
        self.vals_iter = iter(self.vals)
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.num_elements:
            raise StopIteration
        if not self.vals:
            v = []
        else:
            try:
                v = next(self.vals_iter)
            except StopIteration:
                self.vals_iter = iter(self.vals)
                v = next(self.vals_iter)
        chosen_iter_params = dict(zip(self.iter_keys, v))
        self.cfg_flat.update(chosen_iter_params)
        return sample_config(unflatten(self.cfg_flat))

    def __len__(self):
        product = functools.partial(functools.reduce, operator.mul)
        return product(len(v) for v in self.iter_vals) if self.vals else 1


def flatten_dict(d):
    """Flattens a dictionary, ignoring outer keys. Only
    numbers and strings allowed, others will be converted
    to a string."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v))
        elif isinstance(v, (numbers.Number, str, bytes)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def make_validation_dataset(train, n_val, val_length):
    """Partition the training set into training and validation sets.

    Args:
        train (list): List of time series data for training.
        n_val (int): Number of validation samples.
        val_length (int): Length of each validation sample.

    Returns:
        tuple: Lists of training data without validation, validation data, and number of validation samples.
    """
    assert isinstance(train, list), "Train should be a list of series"

    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        print(f"Train length: {len(train_minus_val)}, Val length: {len(val)}")
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)

    return train_minus_val_list, val_list, n_val


def evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn):
    """Evaluate a set of hyperparameters on the validation set.

    Args:
        hyper (dict): Dictionary of hyperparameters to evaluate.
        train_minus_val (list): List of training samples minus validation samples.
        val (list): List of validation samples.
        get_predictions_fn (callable): Function to get predictions.

    Returns:
        float: NLL/D value for the given hyperparameters, averaged over each series.
    """
    assert isinstance(train_minus_val, list) and isinstance(val, list), (
        "Train minus val and val should be lists of series"
    )
    return get_predictions_fn(train_minus_val, val, **hyper, num_samples=0)["NLL/D"]


def convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    elif is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    else:
        return obj


def get_autotuned_predictions_data(
    train,
    test,
    hypers,
    num_samples,
    get_predictions_fn,
    verbose=False,
    parallel=True,
    n_train=None,
    n_val=None,
):
    """
    Automatically tunes hyperparameters based on validation likelihood and retrieves predictions using the best hyperparameters. The validation set is constructed on the fly by splitting the training set.

    Args:
        train (list): List of time series training data.
        test (list): List of time series test data.
        hypers (Union[dict, list]): Either a dictionary specifying the grid search or an explicit list of hyperparameter settings.
        num_samples (int): Number of samples to retrieve.
        get_predictions_fn (callable): Function used to get predictions based on provided hyperparameters.
        verbose (bool, optional): If True, prints out detailed information during the tuning process. Defaults to False.
        parallel (bool, optional): If True, parallelizes the hyperparameter tuning process. Defaults to True.
        n_train (int, optional): Number of training samples to use. Defaults to None.
        n_val (int, optional): Number of validation samples to use. Defaults to None.

    Returns:
        dict: Dictionary containing predictions, best hyperparameters, and other related information.
    """
    if isinstance(hypers, dict):
        hypers = list(grid_iter(hypers))
    else:
        assert isinstance(hypers, list), "hypers must be a list or dict"
    if not isinstance(train, list):
        train = [train]
        test = [test]
    if n_val is None:
        n_val = len(train)
    if len(hypers) > 1:
        val_length = min(
            len(test[0]), int(np.mean([len(series) for series in train]) / 2)
        )
        train_minus_val, val, n_val = make_validation_dataset(
            train, n_val=n_val, val_length=val_length
        )  # use half of train as val for tiny train sets
        # remove validation series that has smaller length than required val_length
        train_minus_val, val = zip(
            *[
                (train_series, val_series)
                for train_series, val_series in zip(train_minus_val, val)
                if len(val_series) == val_length
            ]
        )
        train_minus_val = list(train_minus_val)
        val = list(val)
        if len(train_minus_val) <= int(0.9 * n_val):
            raise ValueError(
                f"Removed too many validation series. Only {len(train_minus_val)} out of {len(n_val)} series have length >= {val_length}. Try or decreasing val_length."
            )
        val_nlls = []

        def eval_hyper(hyper):
            try:
                return hyper, evaluate_hyper(
                    hyper, train_minus_val, val, get_predictions_fn
                )
            except ValueError:
                return hyper, float("inf")

        best_val_nll = float("inf")
        best_hyper = None
        if not parallel:
            for hyper in tqdm(hypers, desc="Hyperparameter search"):
                _, val_nll = eval_hyper(hyper)
                val_nlls.append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_hyper = hyper
                if verbose:
                    print(f"Hyper: {hyper} \n\t Val NLL: {val_nll:3f}")
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_hyper, hyper) for hyper in hypers]
                for future in tqdm(
                    as_completed(futures),
                    total=len(hypers),
                    desc="Hyperparameter search",
                ):
                    hyper, val_nll = future.result()
                    val_nlls.append(val_nll)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_hyper = hyper
                    if verbose:
                        print(f"Hyper: {hyper} \n\t Val NLL: {val_nll:3f}")
    else:
        best_hyper = hypers[0]
        best_val_nll = float("inf")
    print(f"Sampling with best hyper... {best_hyper} \n with NLL {best_val_nll:3f}")
    out = get_predictions_fn(
        train,
        test,
        **best_hyper,
        num_samples=num_samples,
        n_train=n_train,
        parallel=parallel,
    )
    out["best_hyper"] = convert_to_dict(best_hyper)
    return out


@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """

    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x


def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), 0.01)

        def transform(x):
            return x / q

        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return x * q + min_

    return Scaler(transform=transform, inv_transform=inv_transform)


def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.

    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chuncks = input_str.split(settings.time_sep)
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
            # add separator if not already present
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print(
                f"Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}"
            )
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str


def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(
                    f"Warning: Prediction too short {len(pred)} < {expected_length}, returning None"
                )
                return None
            else:
                print(
                    f"Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value"
                )
                return np.concatenate(
                    [pred, np.full(expected_length - len(pred), pred[-1])]
                )
        else:
            return pred[:expected_length]


def generate_predictions(
    completion_fn,
    input_strs,
    steps,
    settings: SerializerSettings,
    num_samples=1,
    temp=0.7,
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs,
):
    """
    Generate and process text completions from a language model for input time series.

    Args:
        completion_fn (callable): Function to obtain text completions from the LLM.
        input_strs (list of array-like): List of input time series.
        steps (int): Number of steps to predict.
        settings (SerializerSettings): Settings for serialization.
        scalers (list of Scaler, optional): List of Scaler objects. Defaults to None, meaning no scaling is applied.
        num_samples (int, optional): Number of samples to return. Defaults to 1.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        parallel (bool, optional): If True, run completions in parallel. Defaults to True.
        strict_handling (bool, optional): If True, return None for predictions that don't have exactly the right format or expected length. Defaults to False.
        max_concurrent (int, optional): Maximum number of concurrent completions. Defaults to 50.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: Tuple containing:
            - preds (list of lists): Numerical predictions.
            - completions_list (list of lists): Raw text completions.
            - input_strs (list of str): Serialized input strings.
    """

    completions_list = []
    complete = lambda x: completion_fn(
        input_str=x,
        steps=steps * STEP_MULTIPLIER,
        settings=settings,
        num_samples=num_samples,
        temp=temp,
    )
    if parallel and len(input_strs) > 1:
        print("Running completions in parallel for each input")
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(
                tqdm(p.map(complete, input_strs), total=len(input_strs))
            )
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

    def completion_to_pred(completion):
        pred = handle_prediction(
            deserialize_str(completion, settings, ignore_last=False, steps=steps),
            expected_length=steps,
            strict=strict_handling,
        )
        if pred is not None:
            return pred
        else:
            return None

    preds = [
        [completion_to_pred(completion) for completion in completions]
        for completions in completions_list
    ]
    return preds, completions_list, input_strs


def get_llmtime_predictions_data(
    train,
    model,
    settings,
    num_samples=10,
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    parallel=True,
    prediction_window: int = 3,
    **kwargs,
):
    """
    Obtain forecasts from an LLM based on training series (history) and evaluate likelihood on test series (true future).
    train and test can be either a single time series or a list of time series.

    Args:
        train (array-like or list of array-like): Training time series data (history).
        model (str): Name of the LLM model to use. Must have a corresponding entry in completion_fns.
        settings (SerializerSettings or dict): Serialization settings.
        num_samples (int, optional): Number of samples to return. Defaults to 10.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        alpha (float, optional): Scaling parameter. Defaults to 0.95.
        beta (float, optional): Shift parameter. Defaults to 0.3.
        basic (bool, optional): If True, use the basic version of data scaling. Defaults to False.
        parallel (bool, optional): If True, run predictions in parallel. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary containing predictions, samples, median, NLL/D averaged over each series, and other related information.
    """

    assert model in completion_fns, (
        f"Invalid model {model}, must be one of {list(completion_fns.keys())}"
    )
    completion_fn = completion_fns[model]

    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    assert isinstance(train, list), "Attention we need a list of 1d numpy arrays"

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))

    test_len = prediction_window

    # transform input_arrs
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = input_arrs
    # serialize input_arrs
    input_strs = [
        serialize_arr(scaled_input_arr, settings)
        for scaled_input_arr in transformed_input_arrs
    ]
    # Truncate input_arrs to fit the maximum context length
    input_arrs, input_strs = zip(
        *[
            truncate_input(input_array, input_str, settings, model, test_len)
            for input_array, input_str in zip(input_arrs, input_strs)
        ]
    )

    steps = prediction_window
    samples = None
    medians = None
    completions_list = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(
            completion_fn,
            input_strs,
            steps,
            settings,
            num_samples=num_samples,
            temp=temp,
            parallel=parallel,
            **kwargs,
        )
        samples = [
            pd.DataFrame(preds[i], columns=[str(i) for i in range(prediction_window)])
            for i in range(len(preds))
        ]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        "samples": samples,
        "median": medians,
        "info": {
            "Method": model,
        },
        "completions_list": completions_list,
        "input_strs": input_strs,
    }
    return out_dict


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LLMTime(BaseLightningModule):
    def __init__(self, model: torch.nn.Module, prediction_window: int = 3, **kwargs):
        super().__init__(**kwargs)

        assert self.experiment_name == "endo_only"

        llma2_hypers = dict(
            temp=0.7,
            alpha=0.95,
            beta=0.3,
            basic=False,
            settings=SerializerSettings(
                base=10, prec=3, signed=True, half_bin_correction=True
            ),
        )

        model_hypers = {
            "LLMA2": {"model": "llama-7b", **llma2_hypers},
        }

        self.hypers = list(grid_iter(model_hypers["LLMA2"]))[0]

        self.automatic_optimization = False

    def model_forward(self, look_back_window: Tensor):
        device = look_back_window.device
        look_back_window = look_back_window.detach().cpu().numpy()
        look_back_window = look_back_window[:, :, 0]  # (B, L)
        look_back_windows = [look_back_window[i] for i in range(len(look_back_window))]
        preds_dict = get_llmtime_predictions_data(look_back_windows, **self.hypers)
        preds = torch.tensor(preds_dict["median"], device=device)  # (B, T)
        return preds

    def train_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    print(torch.cuda.max_memory_allocated())
    print()

    llma2_hypers = dict(
        temp=0.7,
        alpha=0.95,
        beta=0.3,
        basic=False,
        settings=SerializerSettings(
            base=10, prec=3, signed=True, half_bin_correction=True
        ),
    )

    model_hypers = {
        "LLMA2": {"model": "llama-7b", **llma2_hypers},
    }

    model_predict_fns = {
        "LLMA2": get_llmtime_predictions_data,
    }

    model_names = list(model_predict_fns.keys())
    hypers = list(grid_iter(model_hypers["LLMA2"]))

    train = np.zeros((1000,))
    test = np.ones((100,))

    import pdb

    preds = get_llmtime_predictions_data(train, test, **hypers[0])

    pdb.set_trace()
