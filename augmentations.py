import sys
import io
import os
import math
import tqdm
import pathlib
import shutil
import csv
import time
import json
import re
import copy
import collections
import uuid
import logging
import contextlib
from types import MethodType
import augly.audio as audaugs
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert
from torch.utils.data._utils.pin_memory import (
    pin_memory as recursive_pin_memory,
)
from torch.utils.data import (
    RandomSampler,
    WeightedRandomSampler,
    DistributedSampler,
    Sampler,
    IterableDataset,
    DataLoader
)
from packaging import version
import torchaudio
import urllib.request
import pickle
import inspect
from dataclasses import dataclass

# Optional support for webdataset
try:
    import webdataset as wds
    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False


OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"

global OPENRIR_FOLDER

OPENRIR_FOLDER = ""

TORCHAUDIO_FORMATS = ["wav", "flac", "aac", "ogg", "flac", "mp3"]
ITEM_POSTFIX = "_data"

CSVItem = collections.namedtuple("CSVItem", ["data", "format", "opts"])
CSVItem.__doc__ = """The Legacy Extended CSV Data item triplet"""

logger = logging.getLogger(__name__)


"""
    augmentation function must have params - (audio, sample_rate=16000) and return augmented audio
    where is it located? cuda vs cpu
"""


def add_background_noise_white_noise_20db(audio, sample_rate=16000):
    aug_audio, sr = audaugs.add_background_noise(audio.cpu().numpy(), sample_rate=sample_rate, snr_level_db=20)
    return torch.from_numpy(aug_audio).to(audio.device)


def change_volume_up_20db(audio, sample_rate=16000):
    aug_audio, sr = audaugs.change_volume(audio.cpu().numpy(), sample_rate=sample_rate, volume_db=20.0)
    return torch.from_numpy(aug_audio).to(audio.device)


def change_volume_down_20db(audio, sample_rate=16000):
    aug_audio, sr = audaugs.change_volume(audio.cpu().numpy(), sample_rate=sample_rate, volume_db=-20.0)
    return torch.from_numpy(aug_audio).to(audio.device)


def clicks_every_half_second_10db(audio, sample_rate=16000):
    aug_audio, sr = audaugs.clicks(audio.cpu().numpy(), sample_rate=sample_rate, seconds_between_clicks=0.5,
                                   snr_level_db=10.0)
    return torch.from_numpy(aug_audio).to(audio.device)


def pitch_shift_up_2_steps(audio, sample_rate=16000):
    aug_audio, sr = audaugs.pitch_shift(audio.cpu().numpy(), sample_rate=sample_rate, n_steps=2)
    return torch.from_numpy(aug_audio).to(audio.device)


def pitch_shift_down_2_steps(audio, sample_rate=16000):
    aug_audio, sr = audaugs.pitch_shift(audio.cpu().numpy(), sample_rate=sample_rate, n_steps=-2)
    return torch.from_numpy(aug_audio).to(audio.device)


def reverb(audio, sample_rate=16000):
    aug_audio, sr = audaugs.reverb(audio.cpu().numpy(), sample_rate=sample_rate)
    return torch.from_numpy(aug_audio).to(audio.device)


def speed_up_10(audio, sample_rate=16000):
    aug_audio, sr = audaugs.speed(audio.cpu().numpy(), sample_rate=sample_rate, factor=1.1)
    return torch.from_numpy(aug_audio).to(audio.device)


def speed_down_10(audio, sample_rate=16000):
    aug_audio, sr = audaugs.speed(audio.cpu().numpy(), sample_rate=sample_rate, factor=0.9)
    return torch.from_numpy(aug_audio).to(audio.device)


def time_stretch_up_10(audio, sample_rate=16000):
    aug_audio, sr = audaugs.time_stretch(audio.cpu().numpy(), sample_rate=sample_rate, rate=1.1)
    return torch.from_numpy(aug_audio).to(audio.device)


def time_stretch_down_10(audio, sample_rate=16000):
    aug_audio, sr = audaugs.time_stretch(audio.cpu().numpy(), sample_rate=sample_rate, rate=0.9)
    return torch.from_numpy(aug_audio).to(audio.device)


class TimeDomainSpecAugment(torch.nn.Module):
    """A time-domain approximation of the SpecAugment algorithm.

    This augmentation module implements three augmentations in
    the time-domain.

     1. Drop chunks of the audio (zero amplitude or white noise)
     2. Drop frequency bands (with band-drop filters)
     3. Speed peturbation (via resampling to slightly different rate)

    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See ``speechbrain.processing.speech_augmentation.SpeedPerturb``
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).

    Example
    -------
    inputs = torch.randn([10, 16000])
    feature_maker = TimeDomainSpecAugment(speeds=[80])
    feats = feature_maker(inputs, torch.ones(10))
    feats.shape
    torch.Size([10, 12800])
    """

    def __init__(
            self,
            perturb_prob=1.0,
            drop_freq_prob=1.0,
            drop_chunk_prob=1.0,
            speeds=[95, 100, 105],
            sample_rate=16000,
            drop_freq_count_low=0,
            drop_freq_count_high=3,
            drop_chunk_count_low=0,
            drop_chunk_count_high=5,
            drop_chunk_length_low=1000,
            drop_chunk_length_high=2000,
            drop_chunk_noise_factor=0,
    ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, orig_freq=sample_rate, speeds=speeds
        )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high,
        )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor,
        )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        with torch.no_grad():
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return waveforms


class EnvCorrupt(torch.nn.Module):
    """Environmental Corruptions for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).

    Example
    -------
    inputs = torch.randn([10, 16000])
    corrupter = EnvCorrupt(babble_speaker_count=9)
    feats = corrupter(inputs, torch.ones(10))
    """

    def __init__(
            self,
            reverb_prob=1.0,
            babble_prob=1.0,
            noise_prob=1.0,
            openrir_folder=None,
            openrir_max_noise_len=None,
            reverb_csv=None,
            noise_csv=None,
            noise_num_workers=0,
            babble_speaker_count=0,
            babble_snr_low=0,
            babble_snr_high=0,
            noise_snr_low=0,
            noise_snr_high=0,
            rir_scale_factor=1.0,
    ):
        super().__init__()

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):
            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(
                openrir_folder,
                open_reverb_csv,
                open_noise_csv,
                openrir_max_noise_len,
            )

            # Override if they aren't specified
            reverb_csv = reverb_csv or open_reverb_csv
            noise_csv = noise_csv or open_noise_csv

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                rir_scale_factor=rir_scale_factor,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None and noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
            )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms, lengths)
                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms


class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    clean = signal.unsqueeze(0)
    perturbed = perturbator(clean)
    clean.shape
    torch.Size([1, 52173])
    perturbed.shape
    torch.Size([1, 46956])
    """

    def __init__(
            self, orig_freq, speeds=[90, 100, 110], perturb_prob=1.0,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone()

        # Perform a random perturbation
        self.samp_index = torch.randint(len(self.speeds), (1,))[0]
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform


class DropFreq(torch.nn.Module):
    """This class drops a random frequency from the signal.

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    dropper = DropFreq()
    signal = read_audio('samples/audio_samples/example1.wav')
    dropped_signal = dropper(signal.unsqueeze(0))
    """

    def __init__(
            self,
            drop_freq_low=1e-14,
            drop_freq_high=1,
            drop_count_low=1,
            drop_count_high=2,
            drop_width=0.05,
            drop_prob=1,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(
            low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,),
        )

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
                torch.rand(drop_count) * drop_range + self.drop_freq_low
        )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, filter_length, 1, device=waveforms.device)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_width,
            ).to(waveforms.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(torch.nn.Module):
    """This class drops portions of the input signal.

    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    signal = read_audio('samples/audio_samples/example1.wav')
    signal = signal.unsqueeze(0) # [batch, time, channels]
    length = torch.ones(1)
    dropped_signal = dropper(signal, length)
    float(dropped_signal[:, 150])
    0.0
    """

    def __init__(
            self,
            drop_length_low=100,
            drop_length_high=1000,
            drop_count_low=1,
            drop_count_high=10,
            drop_start=0,
            drop_end=None,
            drop_prob=1,
            noise_factor=0.0,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        lengths = (lengths * waveforms.size(1)).long()
        batch_size = waveforms.size(0)
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(batch_size,),
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = torch.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                size=(drop_times[i],),
            )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = torch.randint(
                low=start_min, high=start_max + 1, size=(drop_times[i],),
            )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    dropped_waveform[i, start[j]: end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = torch.rand(length[j], device=waveforms.device)
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, start[j]: end[j]] = noise_vec

        return dropped_waveform


class AddReverb(torch.nn.Module):
    """This class convolves an audio signal with an impulse response.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        impulse response files.
    sorting : str
        The order to iterate the csv file, from one of
        the following options: random, original, ascending, and descending.
    reverb_prob : float
        The chance that the audio signal will be reverbed.
        By default, every batch is reverbed.
    rir_scale_factor: float
        It compresses or dilates the given impulse response.
        If 0 < scale_factor < 1, the impulse response is compressed
        (less reverb), while if scale_factor > 1 it is dilated
        (more reverb).
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.

    Example
    -------
    import pytest
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    clean = signal.unsqueeze(0) # [batch, time, channels]
    reverb = AddReverb('samples/rir_samples/rirs.csv')
    reverbed = reverb(clean, torch.ones(1))
    """

    def __init__(
            self,
            csv_file,
            sorting="random",
            reverb_prob=1.0,
            rir_scale_factor=1.0,
            replacements={},
    ):
        super().__init__()
        self.csv_file = csv_file
        self.sorting = sorting
        self.reverb_prob = reverb_prob
        self.replacements = replacements
        self.rir_scale_factor = rir_scale_factor

        # Create a data loader for the RIR waveforms
        dataset = ExtendedCSVDataset(
            csvpath=self.csv_file,
            sorting=self.sorting if self.sorting != "random" else "original",
            replacements=self.replacements,
        )
        self.data_loader = make_dataloader(
            dataset, shuffle=(self.sorting == "random")
        )
        self.rir_data = iter(self.data_loader)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) > self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Convert length from ratio to number of indices
        # lengths = (lengths * waveforms.shape[1])[:, None, None]

        # Load and prepare RIR
        rir_waveform = self._load_rir(waveforms)

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose(1, -1),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_waveform = rir_waveform.transpose(1, -1)

        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform

    def _load_rir(self, waveforms):
        try:
            rir_waveform, length = next(self.rir_data).at_position(0)
        except StopIteration:
            self.rir_data = iter(self.data_loader)
            rir_waveform, length = next(self.rir_data).at_position(0)

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.type(waveforms.dtype)
        return rir_waveform.to(waveforms.device)


class AddBabble(torch.nn.Module):
    """Simulate babble noise by mixing the signals in a batch.

    Arguments
    ---------
    speaker_count : int
        The number of signals to mix with the original signal.
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    mix_prob : float
        The probability that the batch of signals will be
        mixed with babble noise. By default, every signal is mixed.

    Example
    -------
    import pytest
    babbler = AddBabble()
    dataset = ExtendedCSVDataset(
    ...     csvpath='samples/audio_samples/csv_example3.csv',
    ... )
    loader = make_dataloader(dataset, batch_size=5)
    speech, lengths = next(iter(loader)).at_position(0)
    noisy = babbler(speech, lengths)
    """

    def __init__(
            self, speaker_count=3, snr_low=0, snr_high=0, mix_prob=1,
    ):
        super().__init__()
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            A batch of audio signals to process, with shape `[batch, time]` or
            `[batch, time, channels]`.
        lengths : tensor
            The length of each audio in the batch, with shape `[batch]`.

        Returns
        -------
        Tensor with processed waveforms.
        """

        babbled_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)
        batch_size = len(waveforms)

        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return babbled_waveform

        # Pick an SNR and use it to compute the mixture amplitude factors
        clean_amplitude = compute_amplitude(waveforms, lengths)
        SNR = torch.rand(batch_size, 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= 1 - noise_amplitude_factor

        # For each speaker in the mixture, roll and add
        babble_waveform = waveforms.roll((1,), dims=0)
        babble_len = lengths.roll((1,), dims=0)
        for i in range(1, self.speaker_count):
            babble_waveform += waveforms.roll((1 + i,), dims=0)
            babble_len = torch.max(babble_len, babble_len.roll((1,), dims=0))

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_waveform += babble_waveform

        return babbled_waveform


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    csv_keys : list, None, optional
        Default: None . One data entry for the noise data should be specified.
        If None, the csv file is expected to have only one data entry.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.

    Example
    -------
    import pytest
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    clean = signal.unsqueeze(0) # [batch, time, channels]
    noisifier = AddNoise('samples/noise_samples/noise.csv')
    noisy = noisifier(clean, torch.ones(1))
    """

    def __init__(
            self,
            csv_file=None,
            csv_keys=None,
            sorting="random",
            num_workers=0,
            snr_low=0,
            snr_high=0,
            pad_noise=False,
            mix_prob=1.0,
            start_index=None,
            normalize=False,
            replacements={},
    ):
        super().__init__()

        self.csv_file = csv_file
        self.csv_keys = csv_keys
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.replacements = replacements

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(waveforms)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length = self._load_noise(
                lengths, tensor_length,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
            noisy_waveform += noise_waveform

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = torch.max(
                torch.abs(noisy_waveform), dim=1, keepdim=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        lengths = lengths.long().squeeze(1)
        batch_size = len(lengths)

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = ExtendedCSVDataset(
                    csvpath=self.csv_file,
                    output_keys=self.csv_keys,
                    sorting=self.sorting
                    if self.sorting != "random"
                    else "original",
                    replacements=self.replacements,
                )
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=(self.sorting == "random"),
                )
                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                min_len = torch.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            padding = (0, max_length - noise_batch.size(1))
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(
                high=max_chop, size=(1,), device=lengths.device
            )

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index: start_index + max_length]
        noise_len = (noise_len - start_index).clamp(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(
                noise_batch, noise_lens, added_noise, added_lens
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    @staticmethod
    def _concat_batch(noise_batch, noise_lens, added_noise, added_lens):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]
        pad = (0, abs(noise_tensor_len - added_tensor_len))
        if noise_tensor_len > added_tensor_len:
            added_noise = torch.nn.functional.pad(added_noise, pad)
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = torch.nn.functional.pad(noise_batch, pad)
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len

        noise_batch = torch.cat((noise_batch, added_noise))
        noise_lens = torch.cat((noise_lens, added_lens))

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            noises, lens = next(self.noise_data).at_position(0)
        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens = next(self.noise_data).at_position(0)
        return noises, lens


class Resample(torch.nn.Module):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are
        allowed.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    signal = signal.unsqueeze(0) # [batch, time, channels]
    resampler = Resample(orig_freq=16000, new_freq=8000)
    resampled = resampler(signal)
    signal.shape
    torch.Size([1, 52173])
    resampled.shape
    torch.Size([1, 26087])
    """

    def __init__(
            self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.size()
        window_size = self.weights.size(1)
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = torch.zeros(
            (batch_size, num_channels, tot_output_samp),
            device=waveforms.device,
        )
        self.weights = self.weights.to(waveforms.device)

        # Check weights are on correct device
        if waveforms.device != self.weights.device:
            self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        eye = torch.eye(num_channels, device=waveforms.device).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.size(0)):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = torch.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding)
            )
            conv_wave = torch.nn.functional.conv1d(
                input=wave_to_conv,
                weight=self.weights[i].repeat(num_channels, 1, 1),
                stride=self.conv_stride,
                groups=num_channels,
            )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=self.conv_transpose_stride
            )

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.size(-1)
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding)
            )
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = torch.arange(
            start=0.0, end=self.output_samples, device=waveforms.device,
        )
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * self.orig_freq)
        max_input_index = torch.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = torch.arange(max_weight_width, device=waveforms.device)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (
                1
                + torch.cos(
            2
            * math.pi
            * lowpass_cutoff
            / self.lowpass_filter_width
            * delta_t[inside_window_indices]
        )
        )

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Returns a notch filter constructed from a high-pass and low-pass filter.

    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)

    Arguments
    ---------
    notch_freq : float
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width : int
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width : float
        Width of the notch, as a fraction of the sampling_rate / 2.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    signal = signal.unsqueeze(0).unsqueeze(2)
    kernel = notch_filter(0.25)
    notched_signal = convolve1d(signal, kernel)
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = torch.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return torch.sin(x) / x

        # The zero is at the middle index
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad + 1:])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= torch.blackman_window(filter_width)
    hlpf /= torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= torch.blackman_window(filter_width)
    hhpf /= -torch.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, -1, 1)


def convolve1d(
        waveform,
        kernel,
        padding=0,
        pad_type="constant",
        stride=1,
        groups=1,
        use_fft=False,
        rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution.
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by the number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    from speechbrain.dataio.dataio import read_audio
    signal = read_audio('samples/audio_samples/example1.wav')
    signal = signal.unsqueeze(0).unsqueeze(2)
    kernel = torch.rand(1, 10, 1)
    signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose(2, 1)
    kernel = kernel.transpose(2, 1)

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(
            input=waveform, pad=padding, mode=pad_type,
        )

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(
            kernel.size(0), kernel.size(1), zero_length, device=kernel.device
        )
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        # Multiply in frequency domain to convolve in time domain
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            import torch.fft as fft

            result = fft.rfft(waveform) * fft.rfft(kernel)
            convolved = fft.irfft(result, n=waveform.size(-1))
        else:
            f_signal = torch.rfft(waveform, 1)
            f_kernel = torch.rfft(kernel, 1)
            sig_real, sig_imag = f_signal.unbind(-1)
            ker_real, ker_imag = f_kernel.unbind(-1)
            f_result = torch.stack(
                [
                    sig_real * ker_real - sig_imag * ker_imag,
                    sig_real * ker_imag + sig_imag * ker_real,
                ],
                dim=-1,
            )
            convolved = torch.irfft(
                f_result, 1, signal_sizes=[waveform.size(-1)]
            )

    # Use the implementation given by torch, which should be efficient on GPU
    else:
        convolved = torch.nn.functional.conv1d(
            input=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(2, 1)


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    General function to contaminate a given signal with reverberation given a
    Room Impulse Response (RIR).
    It performs convolution between RIR and signal, but without changing
    the original amplitude of the signal.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    rir_waveform : tensor
        RIR tensor, shape should be [time, channels].
    rescale_amp : str
        Whether reverberated signal is rescaled (None) and with respect either
        to original signal "peak" amplitude or "avg" average amplitude.
        Choose between [None, "avg", "peak"].

    Returns
    -------
    waveforms: tensor
        Reverberated signal.

    """

    orig_shape = waveforms.shape

    if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0).unsqueeze(-1)
    elif len(waveforms.shape) == 2:
        waveforms = waveforms.unsqueeze(-1)

    if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
        rir_waveform = rir_waveform.unsqueeze(0).unsqueeze(-1)
    elif len(rir_waveform.shape) == 2:
        rir_waveform = rir_waveform.unsqueeze(-1)

    # Compute the average amplitude of the clean
    orig_amplitude = compute_amplitude(
        waveforms, waveforms.size(1), rescale_amp
    )

    # Compute index of the direct signal, so we can preserve alignment
    value_max, direct_index = rir_waveform.abs().max(axis=1, keepdim=True)

    # Making sure the max is always positive (if not, flip)
    # mask = torch.logical_and(rir_waveform == value_max,  rir_waveform < 0)
    # rir_waveform[mask] = -rir_waveform[mask]

    # Use FFT to compute convolution, because of long reverberation filter
    waveforms = convolve1d(
        waveform=waveforms,
        kernel=rir_waveform,
        use_fft=True,
        rotation_index=direct_index,
    )

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(
        waveforms, waveforms.size(1), orig_amplitude, rescale_amp
    )

    if len(orig_shape) == 1:
        waveforms = waveforms.squeeze(0).squeeze(-1)
    if len(orig_shape) == 2:
        waveforms = waveforms.squeeze(-1)

    return waveforms


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
    """This functions performs signal rescaling to a target level.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    target_lvl : float
        Target lvl in dB or linear scale.
    amp_type : str
        Whether one wants to rescale with respect to "avg" or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        whether target_lvl belongs to linear or dB scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    waveforms : tensor
        Rescaled waveforms.
    """

    assert amp_type in ["peak", "avg"]
    assert scale in ["linear", "dB"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    waveforms = normalize(waveforms, lengths, amp_type)

    if scale == "linear":
        out = target_lvl * waveforms
    elif scale == "dB":
        out = dB_to_amplitude(target_lvl) * waveforms

    else:
        raise NotImplementedError("Invalid scale, choose between dB and linear")

    if batch_added:
        out = out.squeeze(0)

    return out


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    round(dB_to_amplitude(SNR=10), 3)
    3.162
    dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    """This function normalizes a signal to unitary average or peak amplitude.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether one wants to normalize with respect to "avg" or "peak"
        amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
        is not prevented and can occur.
    eps : float
        A small number to add to the denominator to prevent NaN.

    Returns
    -------
    waveforms : tensor
        Normalized level waveform.
    """

    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den

def _prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
    """Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        logger.info("Skipping OPENRIR download and extract, path found")
        # download_file(OPENRIR_URL, filepath)
    else:
        download_file(OPENRIR_URL, filepath, unpack=True)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    try:
        if if_main_process():
            with open(csv_file, "w") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in open(filelist):

                    # Read file for duration/channel info
                    filename = os.path.join(folder, line.split()[-1])
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:
                        # Delete old file
                        os.remove(filename)
                        for i in range(int(duration / max_length)):
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = (
                                filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                f"{ID}_{i}",
                                str((stop - start) / rate),
                                new_filename,
                                ext,
                                "\n",
                            )
                            w.write(",".join(csv_row))
                    else:
                        w.write(
                            ",".join((ID, str(duration), filename, ext, "\n"))
                        )
    finally:
        ddp_barrier()


def download_file(
    source, dest, unpack=False, dest_unpack=None, replace_existing=False
):
    """Downloads the file from the given source and saves it in the given
    destination path.

     Arguments
    ---------
    source : path or url
        Path of the source file. If the source is an URL, it downloads it from
        the web.
    dest : path
        Destination path.
    unpack : bool
        If True, it unpacks the data in the dest folder.
    replace_existing : bool
        If True, replaces the existing files.
    """
    try:
        if if_main_process():

            class DownloadProgressBar(tqdm.tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            # Create the destination directory if it doesn't exist
            dest_dir = pathlib.Path(dest).resolve().parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            if "http" not in source:
                shutil.copyfile(source, dest)

            elif not os.path.isfile(dest) or (
                os.path.isfile(dest) and replace_existing
            ):
                print(f"Downloading {source} to {dest}")
                with DownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc=source.split("/")[-1],
                ) as t:
                    urllib.request.urlretrieve(
                        source, filename=dest, reporthook=t.update_to
                    )
            else:
                print(f"{dest} exists. Skipping download")

            # Unpack if necessary
            if unpack:
                if dest_unpack is None:
                    dest_unpack = os.path.dirname(dest)
                print(f"Extracting {dest} to {dest_unpack}")
                shutil.unpack_archive(dest, dest_unpack)
    finally:
        ddp_barrier()


class DynamicItemDataset(Dataset):
    """Dataset that reads, wrangles, and produces dicts.

    Each data point dict provides some items (by key), for example, a path to a
    wavefile with the key "wav_file". When a data point is fetched from this
    Dataset, more items are produced dynamically, based on pre-existing items
    and other dynamic created items. For example, a dynamic item could take the
    wavfile path and load the audio from the disk.

    The dynamic items can depend on other dynamic items: a suitable evaluation
    order is used automatically,  as long as there are no circular dependencies.

    A specified list of keys is collected in the output dict. These can be items
    in the original data or dynamic items. If some dynamic items are not
    requested, nor depended on by other requested items, they won't be computed.
    So for example if a user simply wants to iterate over the text, the
    time-consuming audio loading can be skipped.

    About the format:
    Takes a dict of dicts as the collection of data points to read/wrangle.
    The top level keys are data point IDs.
    Each data point (example) dict should have the same keys, corresponding to
    different items in that data point.

    Altogether the data collection could look like this:

    data = {
    ...  "spk1utt1": {
    ...      "wav_file": "/path/to/spk1utt1.wav",
    ...      "text": "hello world",
    ...      "speaker": "spk1",
    ...      },
    ...  "spk1utt2": {
    ...      "wav_file": "/path/to/spk1utt2.wav",
    ...      "text": "how are you world",
    ...      "speaker": "spk1",
    ...      }
    ... }

    NOTE
    ----
        The top-level key, the data point id, is implicitly added as an item
        in the data point, with the key "id"

    Each dynamic item is configured by three things: a key, a func, and a list
    of argkeys. The key should be unique among all the items (dynamic or not) in
    each data point. The func is any callable, and it returns the dynamic item's
    value. The callable is called with the values of other items as specified
    by the argkeys list (as positional args, passed in the order specified by
    argkeys).

    The dynamic_items configuration could look like this:

    import torch
    dynamic_items = [
    ...     {"func": lambda l: torch.Tensor(l),
    ...     "takes": ["wav_loaded"],
    ...     "provides": "wav"},
    ...     {"func": lambda path: [ord(c)/100 for c in path],  # Fake "loading"
    ...     "takes": ["wav_file"],
    ...     "provides": "wav_loaded"},
    ...     {"func": lambda t: t.split(),
    ...     "takes": ["text"],
    ...     "provides": "words"}]

    With these, different views of the data can be loaded:

    from speechbrain.dataio.dataloader import SaveableDataLoader
    from speechbrain.dataio.batch import PaddedBatch
    dataset = DynamicItemDataset(data, dynamic_items)
    dataloader = SaveableDataLoader(dataset, collate_fn=PaddedBatch,
    ...     batch_size=2)
    # First, create encoding for words:
    dataset.set_output_keys(["words"])
    encoding = {}
    next_id = 1
    for batch in dataloader:
    ...     for sent in batch.words:
    ...         for word in sent:
    ...             if word not in encoding:
    ...                 encoding[word] = next_id
    ...                 next_id += 1
    # Next, add an encoded words_tensor dynamic item:
    dataset.add_dynamic_item(
    ...     func = lambda ws: torch.tensor([encoding[w] for w in ws],
    ...             dtype=torch.long),
    ...     takes = ["words"],
    ...     provides = "words_encoded")
    # Now we can get word and audio tensors:
    dataset.set_output_keys(["id", "wav", "words_encoded"])
    batch = next(iter(dataloader))
    batch.id
    ['spk1utt1', 'spk1utt2']
    batch.wav  # +ELLIPSIS
    PaddedData(data=tensor([[0.4700, 1.1200, ...
    batch.words_encoded
    PaddedData(data=tensor([[1, 2, 0, 0],
            [3, 4, 5, 2]]), lengths=tensor([0.5000, 1.0000]))

    Output keys can also be a map:

    dataset.set_output_keys({"id":"id", "signal": "wav", "words": "words_encoded"})
    batch = next(iter(dataloader))
    batch.words
    PaddedData(data=tensor([[1, 2, 0, 0],
            [3, 4, 5, 2]]), lengths=tensor([0.5000, 1.0000]))


    Arguments
    ---------
    data : dict
        Dictionary containing single data points (e.g. utterances).
    dynamic_items : list, optional
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
    output_keys : dict, list, optional
        List of keys (either directly available in data or dynamic items)
        to include in the output dict when data points are fetched.

        If a dict is given; it is used to map internal keys to output keys.
        From the output_keys dict key:value pairs the key appears outside,
        and value is the internal key.
    """

    def __init__(
        self, data, dynamic_items=[], output_keys=[],
    ):
        self.data = data
        self.data_ids = list(self.data.keys())
        static_keys = list(self.data[self.data_ids[0]].keys())
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
        self.pipeline = DataPipeline(static_keys, dynamic_items)
        self.set_output_keys(output_keys)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        return self.pipeline.compute_outputs({"id": data_id, **data_point})

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Makes a new dynamic item available on the dataset.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item).
        But otherwise, should use:
        add_dynamic_item(func, takes, provides).

        See `speechbrain.utils.data_pipeline`.

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        provides : str
            Unique key or keys that this provides.
        """
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        These are the keys that are actually evaluated when a data point
        is fetched from the dataset.

        Arguments
        ---------
        keys : dict, list
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.pipeline.set_output_keys(keys)

    @contextlib.contextmanager
    def output_keys_as(self, keys):
        """Context manager to temporarily set output keys.

        Example
        -------
        dataset = DynamicItemDataset({"a":{"x":1,"y":2},"b":{"x":3,"y":4}},
        ...     output_keys = ["x"])
        with dataset.output_keys_as(["y"]):
        ...     print(dataset[0])
        {'y': 2}
        print(dataset[0])
        {'x': 1}

        NOTE
        ----
        Not thread-safe. While in this context manager, the output keys
        are affected for any call.
        """
        saved_output = self.pipeline.output_mapping
        self.pipeline.set_output_keys(keys)
        yield self
        self.pipeline.set_output_keys(saved_output)

    def filtered_sorted(
        self,
        key_min_value={},
        key_max_value={},
        key_test={},
        sort_key=None,
        reverse=False,
        select_n=None,
    ):
        """Get a filtered and/or sorted version of this, shares static data.

        The reason to implement these operations in the same method is that
        computing some dynamic items may be expensive, and this way the
        filtering and sorting steps don't need to compute the dynamic items
        twice.

        Arguments
        ---------
        key_min_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] >= limit
        key_max_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] <= limit
        key_test : dict
            Map from key (in data or in dynamic items) to func, will only keep
            data_point if bool(func(data_point[key])) == True
        sort_key : None, str
            If not None, sort by data_point[sort_key]. Default is ascending
            order.
        reverse : bool
            If True, sort in descending order.
        select_n : None, int
            If not None, only keep (at most) the first n filtered data_points.
            The possible sorting is applied, but only on the first n data
            points found. Meant for debugging.

        Returns
        -------
        FilteredSortedDynamicItemDataset
            Shares the static data, but has its own output keys and
            dynamic items (initially deep copied from this, so they have the
            same dynamic items available)

        NOTE
        ----
        Temporarily changes the output keys!
        """
        filtered_sorted_ids = self._filtered_sorted_ids(
            key_min_value, key_max_value, key_test, sort_key, reverse, select_n,
        )
        return FilteredSortedDynamicItemDataset(
            self, filtered_sorted_ids
        )  # NOTE: defined below

    def _filtered_sorted_ids(
        self,
        key_min_value={},
        key_max_value={},
        key_test={},
        sort_key=None,
        reverse=False,
        select_n=None,
    ):
        """Returns a list of data ids, fulfilling the sorting and filtering."""

        def combined_filter(computed):
            for key, limit in key_min_value.items():
                # NOTE: docstring promises >= so using that.
                # Mathematically could also use < for nicer syntax, but
                # maybe with some super special weird edge case some one can
                # depend on the >= operator
                if computed[key] >= limit:
                    continue
                return False
            for key, limit in key_max_value.items():
                if computed[key] <= limit:
                    continue
                return False
            for key, func in key_test.items():
                if bool(func(computed[key])):
                    continue
                return False
            return True

        temp_keys = (
            set(key_min_value.keys())
            | set(key_max_value.keys())
            | set(key_test.keys())
            | set([] if sort_key is None else [sort_key])
        )
        filtered_ids = []
        with self.output_keys_as(temp_keys):
            for i, data_id in enumerate(self.data_ids):
                if select_n is not None and len(filtered_ids) == select_n:
                    break
                data_point = self.data[data_id]
                data_point["id"] = data_id
                computed = self.pipeline.compute_outputs(data_point)
                if combined_filter(computed):
                    if sort_key is not None:
                        # Add (main sorting index, current index, data_id)
                        # So that we maintain current sorting and don't compare
                        # data_id values ever.
                        filtered_ids.append((computed[sort_key], i, data_id))
                    else:
                        filtered_ids.append(data_id)
        if sort_key is not None:
            filtered_sorted_ids = [
                tup[2] for tup in sorted(filtered_ids, reverse=reverse)
            ]
        else:
            filtered_sorted_ids = filtered_ids
        return filtered_sorted_ids

    @classmethod
    def from_json(
        cls, json_path, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Load a data prep JSON file and create a Dataset based on it."""
        data = load_data_json(json_path, replacements)
        return cls(data, dynamic_items, output_keys)

    @classmethod
    def from_csv(
        cls, csv_path, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Load a data prep CSV file and create a Dataset based on it."""
        data = load_data_csv(csv_path, replacements)
        return cls(data, dynamic_items, output_keys)

    @classmethod
    def from_arrow_dataset(
        cls, dataset, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Loading a prepared huggingface dataset"""
        # define an unbound method to generate puesdo keys
        def keys(self):
            return [i for i in range(dataset.__len__())]

        # bind this method to arrow dataset
        dataset.keys = MethodType(keys, dataset)
        return cls(dataset, dynamic_items, output_keys)


class ExtendedCSVDataset(DynamicItemDataset):
    """Extended CSV compatibility for DynamicItemDataset.

    Uses the SpeechBrain Extended CSV data format, where the CSV must have an
    'ID' and 'duration' fields.

    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``

    These add a <name>_sb_data item in the dict. Additionally, a basic
    DynamicItem (see DynamicItemDataset) is created, which loads the _sb_data
    item.

    Bash-like string replacements with $to_replace are supported.

    NOTE
    ----
    Mapping from legacy interface:

    - csv_file -> csvpath
    - sentence_sorting -> sorting, and "random" is not supported, use e.g.
      ``make_dataloader(..., shuffle = (sorting=="random"))``
    - avoid_if_shorter_than -> min_duration
    - avoid_if_longer_than -> max_duration
    - csv_read -> output_keys, and if you want IDs add "id" as key

    Arguments
    ---------
    csvpath : str, path
        Path to extended CSV.
    replacements : dict
        Used for Bash-like $-prefixed substitution,
        e.g. ``{"data_folder": "/home/speechbrain/data"}``, which would
        transform `$data_folder/utt1.wav` into `/home/speechbain/data/utt1.wav`
    sorting : {"original", "ascending", "descending"}
        Keep CSV order, or sort ascending or descending by duration.
    min_duration : float, int
        Minimum duration in seconds. Discards other entries.
    max_duration : float, int
        Maximum duration in seconds. Discards other entries.
    dynamic_items : list
        Configuration for extra dynamic items produced when fetching an
        example. List of DynamicItems or dicts with keys::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
        NOTE: A dynamic item is automatically added for each CSV data-triplet
    output_keys : list, None
        The list of output keys to produce. You can refer to the names of the
        CSV data-triplets. E.G. if the CSV has: wav,wav_format,wav_opts,
        then the Dataset has a dynamic item output available with key ``"wav"``
        NOTE: If None, read all existing.
    """

    def __init__(
        self,
        csvpath,
        replacements={},
        sorting="original",
        min_duration=0,
        max_duration=36000,
        dynamic_items=[],
        output_keys=[],
    ):
        if sorting not in ["original", "ascending", "descending"]:
            clsname = self.__class__.__name__
            raise ValueError(f"{clsname} doesn't support {sorting} sorting")
        # Load the CSV, init class
        data, di_to_add, data_names = load_sb_extended_csv(
            csvpath, replacements
        )
        super().__init__(data, dynamic_items, output_keys)
        self.pipeline.add_dynamic_items(di_to_add)
        # Handle filtering, sorting:
        reverse = False
        sort_key = None
        if sorting == "ascending" or "descending":
            sort_key = "duration"
        if sorting == "descending":
            reverse = True
        filtered_sorted_ids = self._filtered_sorted_ids(
            key_min_value={"duration": min_duration},
            key_max_value={"duration": max_duration},
            sort_key=sort_key,
            reverse=reverse,
        )
        self.data_ids = filtered_sorted_ids
        # Handle None output_keys (differently than Base)
        if not output_keys:
            self.set_output_keys(data_names)


class FilteredSortedDynamicItemDataset(DynamicItemDataset):
    """Possibly filtered, possibly sorted DynamicItemDataset.

    Shares the static data (reference).
    Has its own dynamic_items and output_keys (deepcopy).
    """

    def __init__(self, from_dataset, data_ids):
        self.data = from_dataset.data
        self.data_ids = data_ids
        self.pipeline = copy.deepcopy(from_dataset.pipeline)

    @classmethod
    def from_json(
        cls, json_path, replacements={}, dynamic_items=None, output_keys=None
    ):
        raise TypeError("Cannot create SubsetDynamicItemDataset directly!")

    @classmethod
    def from_csv(
        cls, csv_path, replacements={}, dynamic_items=None, output_keys=None
    ):
        raise TypeError("Cannot create SubsetDynamicItemDataset directly!")


def load_data_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values.

    Arguments
    ----------
    json_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}.
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        JSON data with replacements applied.

    Example
    -------
    json_spec = '''{
    ...   "ex1": {"files": ["{ROOT}/mic1/ex1.wav", "{ROOT}/mic2/ex1.wav"], "id": 1},
    ...   "ex2": {"files": [{"spk1": "{ROOT}/ex2.wav"}, {"spk2": "{ROOT}/ex2.wav"}], "id": 2}
    ... }
    ... '''
    tmpfile = getfixture('tmpdir') / "test.json"
    with open(tmpfile, "w") as fo:
    ...     _ = fo.write(json_spec)
    data = load_data_json(tmpfile, {"ROOT": "/home"})
    data["ex1"]["files"][0]
    '/home/mic1/ex1.wav'
    data["ex2"]["files"][1]["spk2"]
    '/home/ex2.wav'

    """
    with open(json_path, "r") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json


def _recursive_format(data, replacements):
    # Data: dict or list, replacements : dict
    # Replaces string keys in replacements by their values
    # at all levels of data (in str values)
    # Works in-place.
    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[key] = item.format_map(replacements)
            # If not dict, list or str, do nothing
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[i] = item.format_map(replacements)
            # If not dict, list or str, do nothing


def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    Uses the SpeechBrain legacy CSV data format, where the CSV must have an
    'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.

    Example
    -------
    csv_spec = '''ID,duration,wav_path
    ... utt1,1.45,$data_folder/utt1.wav
    ... utt2,2.0,$data_folder/utt2.wav
    ... '''
    tmpfile = getfixture("tmpdir") / "test.csv"
    with open(tmpfile, "w") as fo:
    ...     _ = fo.write(csv_spec)
    data = load_data_csv(tmpfile, {"data_folder": "/home"})
    data["utt1"]["wav_path"]
    '/home/utt1.wav'
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def load_sb_extended_csv(csv_path, replacements={}):
    """Loads SB Extended CSV and formats string values.

    Uses the SpeechBrain Extended CSV data format, where the
    CSV must have an 'ID' and 'duration' fields.

    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``.

    These add a <name>_sb_data item in the dict. Additionally, a
    basic DynamicItem (see DynamicItemDataset) is created, which
    loads the _sb_data item.

    Bash-like string replacements with $to_replace are supported.

    This format has its restriction, but they allow some tasks to
    have loading specified by the CSV.

    Arguments
    ----------
    csv_path : str
        Path to the CSV file.
    replacements : dict
        Optional dict:
        e.g. ``{"data_folder": "/home/speechbrain/data"}``
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    list
        List of DynamicItems to add in DynamicItemDataset.

    """
    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        if not reader.fieldnames[0] == "ID":
            raise KeyError(
                "CSV has to have an 'ID' field, with unique ids"
                " for all data points"
            )
        if not reader.fieldnames[1] == "duration":
            raise KeyError(
                "CSV has to have an 'duration' field, "
                "with the length of the data point in seconds."
            )
        if not len(reader.fieldnames[2:]) % 3 == 0:
            raise ValueError(
                "All named fields must have 3 entries: "
                "<name>, <name>_format, <name>_opts"
            )
        names = reader.fieldnames[2::3]
        for row in reader:
            # Make a triplet for each name
            data_point = {}
            # ID:
            data_id = row["ID"]
            del row["ID"]  # This is used as a key in result, instead.
            # Duration:
            data_point["duration"] = float(row["duration"])
            del row["duration"]  # This is handled specially.
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            # Only need to run these in the actual data,
            # not in _opts, _format
            for key, value in list(row.items())[::3]:
                try:
                    row[key] = variable_finder.sub(
                        lambda match: replacements[match[1]], value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            for i, name in enumerate(names):
                triplet = CSVItem(*list(row.values())[i * 3 : i * 3 + 3])
                data_point[name + ITEM_POSTFIX] = triplet
            result[data_id] = data_point
        # Make a DynamicItem for each CSV entry
        # _read_csv_item delegates reading to further
        dynamic_items_to_add = []
        for name in names:
            di = {
                "func": _read_csv_item,
                "takes": name + ITEM_POSTFIX,
                "provides": name,
            }
            dynamic_items_to_add.append(di)
        return result, dynamic_items_to_add, names


def _read_csv_item(item):
    """Reads the different formats supported in SB Extended CSV.

    Delegates to the relevant functions.
    """
    opts = _parse_csv_item_opts(item.opts)
    if item.format in TORCHAUDIO_FORMATS:
        audio, _ = torchaudio.load(item.data)
        return audio.squeeze(0)
    elif item.format == "pkl":
        return read_pkl(item.data, opts)
    elif item.format == "string":
        # Just implement string reading here.
        # NOTE: No longer supporting
        # lab2ind mapping like before.
        # Try decoding string
        string = item.data
        try:
            string = string.decode("utf-8")
        except AttributeError:
            pass
        # Splitting elements with ' '
        string = string.split(" ")
        return string
    else:
        raise TypeError(f"Don't know how to read {item.format}")


def _parse_csv_item_opts(entry):
    """Parse the _opts field in a SB Extended CSV item."""
    # Accepting even slightly weirdly formatted entries:
    entry = entry.strip()
    if len(entry) == 0:
        return {}
    opts = {}
    for opt in entry.split(" "):
        opt_name, opt_val = opt.split(":")
        opts[opt_name] = opt_val
    return opts


def read_pkl(file, data_options={}, lab2ind=None):
    """This function reads tensors store in pkl format.

    Arguments
    ---------
    file : str
        The path to file to read.
    data_options : dict, optional
        A dictionary containing options for the reader.
    lab2ind : dict, optional
        Mapping from label to integer indices.

    Returns
    -------
    numpy.array
        The array containing the read signal.
    """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except pickle.UnpicklingError:
        err_msg = "cannot read the pkl file %s" % (file)
        raise ValueError(err_msg)

    type_ok = False

    if isinstance(pkl_element, list):

        if isinstance(pkl_element[0], float):
            tensor = torch.FloatTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], int):
            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], str):

            # convert string to integer as specified in self.label_dict
            if lab2ind is not None:
                for index, val in enumerate(pkl_element):
                    pkl_element[index] = lab2ind[val]

            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if not (type_ok):
            err_msg = (
                "The pkl file %s can only contain list of integers, "
                "floats, or strings. Got %s"
            ) % (file, type(pkl_element[0]))
            raise ValueError(err_msg)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


class DynamicItem:
    """Essentially represents a data transformation function.

    A DynamicItem takes some arguments and computes its value dynamically when
    called. A straight-forward use-case is to load something from disk
    dynamically; take the path and provide the loaded data.

    Instances of this class are often created implicitly via the
    @takes and @provides decorators or otherwise from specifying the taken and
    provided arguments and the function.

    A counterpart is the GeneratorDynamicItem, which should be used for
    generator functions.

    Arguments
    ---------
    takes : list
        The keys of the items that this needs to compute its output.
    func : callable
        The function that is used to compute the output.
    provides : list
        The keys that this provides.
    """

    def __init__(self, takes=[], func=None, provides=[]):
        self.takes = takes
        self.func = func
        self.provides = provides

    def __call__(self, *args):
        return self.func(*args)

    # The next methods are more about supporting GeneratorDynamicItems
    def next_takes(self):
        """The next argkeys to provide to this, when called."""
        # Regular function DynamicItems always just need the same set of args
        return self.takes

    def next_provides(self):
        """The next keys that this provides, when called."""
        # Regular function DynamicItems always just provide the same set of keys
        return self.provides

    def provided_in_order(self):
        """Assuming that this may need to be called multiple times; which keys
        does it provide at that call. Returns a list, with len equal to the
        number of times that this may be called."""
        # Regular function DynamicItems are only called once:
        return [self.provides]

    def reset(self):
        """Signals that this will not be called any more times on this pipeline
        call."""
        # Regular function DynamicItems don't need special resets.
        pass

def takes(*argkeys):
    """Decorator which makes a DynamicItem and specifies its argkeys.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the argkeys for that. Otherwise creates a new regular
    DynamicItem, with argkeys specified.

    The args are always passed to the function at the start. Generators could
    support sending new arguments, but for such use cases, simply create a new
    dynamic item. The GeneratorDynamicItem class is meant for pipelines which
    take in an input and transform it in multiple ways, where the intermediate
    representations may be needed for e.g. fitting a BPE segmenter.

    Example
    -------
    @takes("text")
    ... def tokenize(text):
    ...     return text.strip().lower().split()
    tokenize.provides = ["tokenized"]
    tokenize('\tThis Example gets tokenized')
    ['this', 'example', 'gets', 'tokenized']
    """

    def decorator(obj):
        if isinstance(obj, DynamicItem):
            if obj.takes:
                raise ValueError("Can't overwrite DynamicItem.takes")
            obj.takes = argkeys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(takes=argkeys, func=obj)
        else:
            return DynamicItem(takes=argkeys, func=obj)

    return decorator


class GeneratorDynamicItem(DynamicItem):
    """Essentially represents a multi-step data transformation.

    This is the generator function counterpart for DynamicItem (which should be
    used for regular functions).

    A GeneratorDynamicItem first takes some arguments and then uses those in
    multiple steps to incrementally compute some values when called.

    A typical use-case is a pipeline of transformations on data: e.g. taking in
    text as a string, and first a tokenized version, and then on the second
    call providing an integer-encoded version. This can be used even though the
    integer-encoder needs to be trained on the first outputs.

    The main benefit is to be able to define the pipeline in a clear function,
    even if parts of the pipeline depend on others for their initialization.

    Example
    -------
    lab2ind = {}
    def text_pipeline(text):
    ...     text = text.lower().strip()
    ...     text = "".join(c for c in text if c.isalpha() or c == " ")
    ...     words = text.split()
    ...     yield words
    ...     encoded = [lab2ind[word] for word in words]
    ...     yield encoded
    item = GeneratorDynamicItem(
    ...         func=text_pipeline,
    ...         takes=["text"],
    ...         provides=["words", "words_encoded"])
    # First create the integer-encoding:
    ind = 1
    for token in item("Is this it? - This is it."):
    ...     if token not in lab2ind:
    ...         lab2ind[token] = ind
    ...         ind += 1
    # Now the integers can be encoded!
    item()
    [1, 2, 3, 2, 1, 3]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Doesn't generate electricity, only stores the currently active
        # generator:
        self.current_generator = None
        self.num_provided_items = 0

    def __call__(self, *args):
        if self.num_provided_items == len(self.provides):
            raise RuntimeError("DynamicItemPipeline called too many times!")
        if not self.current_generator:
            self.current_generator = self.func(*args)
        # NOTE: Not supporting sending new values to the pipeline.
        out = next(self.current_generator)
        self.num_provided_items += 1
        return out

    def next_takes(self):
        if not self.current_generator:
            return self.takes
        else:
            return []

    def next_provides(self):
        keys = self.provides[self.num_provided_items]
        # Support multiple yielded values like:
        # @yields("wav_read", ["left_ch", "right_ch"])
        if isinstance(keys, str):
            return [keys]
        else:
            return keys

    def provided_in_order(self):
        in_order = []
        for keys in self.provides:
            # Support multiple yielded values like:
            # @provides("wav_read", ["left_ch", "right_ch"])
            if isinstance(keys, str):
                in_order.append([keys])
            else:
                in_order.append(keys)
        return in_order

    def reset(self):
        if self.current_generator is not None:
            self.current_generator.close()
        self.current_generator = None
        self.num_provided_items = 0


def provides(*output_keys):
    """Decorator which makes a DynamicItem and specifies what keys it provides.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the provided keys for that. Otherwise creates a new regular
    DynamicItem, with provided keys specified.

    NOTE
    ----
    The behavior is slightly different for generators and regular functions, if
    many output keys are specified, e.g. @provides("signal", "mfcc"). Regular
    functions should return a tuple with len equal to len(output_keys), while
    generators should yield the items one by one.

    @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     feat = [s**2 for s in wav]
    ...     return wav, feat
    @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     yield wav
    ...     feat = [s**2 for s in wav]
    ...     yield feat

    If multiple keys are yielded at once, write e.g.,

    @provides("wav_read", ["left_channel", "right_channel"])
    ... def read_multi_channel():
    ...     wav = [[.1,.2,-.1],[.2,.1,-.1]]
    ...     yield wav
    ...     yield wav[0], wav[1]

    """

    def decorator(obj):
        if isinstance(obj, DynamicItem):
            if obj.provides:
                raise ValueError("Can't overwrite DynamicItem provides-list.")
            obj.provides = output_keys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(func=obj, provides=output_keys)
        else:
            return DynamicItem(func=obj, provides=output_keys)

    return decorator


takes_decorator = takes  # Just for DataPipeline.add_dynamic_item
provides_decorator = provides  # Just for DataPipeline.add_dynamic_item


class DataPipeline:
    """Organises data transformations into a pipeline.

    Example
    -------
    pipeline = DataPipeline(
    ...     static_data_keys=["text"],
    ...     dynamic_items=[
    ...     {"func": lambda x: x.lower(), "takes": "text", "provides": "foo"},
    ...     {"func": lambda x: x[::-1], "takes": "foo", "provides": "bar"},
    ...     ],
    ...     output_keys=["bar"],
    ... )
    pipeline({"text": "Test"})
    {'bar': 'tset'}
    """

    def __init__(self, static_data_keys, dynamic_items=[], output_keys=[]):
        self.dg = DependencyGraph()
        self._exec_order = None
        self.key_to_node = {}
        self.unaccounted_keys = {}
        self.dynamic_items = []
        self.output_mapping = {}
        self.add_static_keys(static_data_keys)
        self.add_dynamic_items(dynamic_items)
        self.set_output_keys(output_keys)

    def add_static_keys(self, static_keys):
        """Informs the pipeline about static items.

        Static items are the ones provided to __call__ as data.
        """
        for key in static_keys:
            node_id = self.dg.add_node(data=StaticItem(key=key))
            self.key_to_node[key] = node_id

    def add_dynamic_items(self, dynamic_items):
        """Add multiple dynamic items at once."""
        for item in dynamic_items:
            try:
                self.add_dynamic_item(**item)
            except TypeError:
                self.add_dynamic_item(item)

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Adds a dynamic item to the Pipeline.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item)
        But otherwise, should use:
        add_dynamic_item(func, takes, provides)

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single key can be given as a bare string.
        provides : str, list
            For regular functions, the key or list of keys that it provides.
            If you give a generator function, key or list of keys that it
            yields, in order. Also see the provides decorator.
            A single key can be given as a bare string.
        """
        if isinstance(func, DynamicItem):
            if takes is not None or provides is not None:
                raise ValueError(
                    "If providing a DynamicItem directly, don't "
                    "specify takes or provides"
                )
            else:
                self._add_dynamic_item_object(func)
                return
        if isinstance(takes, str):
            takes = [takes]
        if isinstance(provides, str):
            provides = [provides]
        di = takes_decorator(*takes)(provides_decorator(*provides)(func))
        self._add_dynamic_item_object(di)

    def _add_dynamic_item_object(self, obj):
        """Internally adds the object.

        There is a node in the dependency graph for each call of the
        DynamicItem. Each call may return multiple keys and depend on multiple
        keys. An internal dict maps key to the id of the node that produces it.
        """
        if not obj.provides:
            raise ValueError(
                "Won't add redundant dynamic item which doesn't "
                "provide anything."
            )
        depended = []
        for key in obj.takes:
            # Might not be accounted for, yet:
            if key not in self.key_to_node:
                dependee_keys = self.unaccounted_keys.setdefault(key, [])
                dependee_keys.extend(obj.next_provides())
            else:
                depended.append(self.key_to_node[key])
        for provided in obj.provided_in_order():
            node_id = self.dg.add_node(data=obj)
            for key in provided:
                self.key_to_node[key] = node_id
                # This key may also be unaccounted for, so account for it now:
                if key in self.unaccounted_keys:
                    for dependee_key in self.unaccounted_keys[key]:
                        dependee_node = self.key_to_node[dependee_key]
                        self.dg.add_edge(dependee_node, node_id)
                    del self.unaccounted_keys[key]  # Now accounted for!
            for dep_id in depended:
                self.dg.add_edge(node_id, dep_id)
            # Next call will depend on this call:
            depended = [node_id]
        # Keep a reference to the item in this object, as well:
        self.dynamic_items.append(obj)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        Also re-evaluates execution order.
        So if you request different outputs, some parts of the
        data pipeline may be skipped.

        Arguments
        ---------
        keys : dict, list, None
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.output_mapping = self._output_keys_to_mapping(keys)
        self._exec_order = None

    @staticmethod
    def _output_keys_to_mapping(keys):
        # Ensure a mapping (accept a list for convenience, too)
        if keys is None:
            output_mapping = {}
        elif isinstance(keys, dict):
            output_mapping = keys
        else:
            output_mapping = {key: key for key in keys}
        return output_mapping

    def compute_outputs(self, data):
        """
        Arguments
        ---------
        data : dict
            Dictionary with data entries by key.

        Returns
        -------
        dict
            With the keys that were set.
        """
        if self._exec_order is None:
            self._prepare_run(data)
        return self._compute(data, self._exec_order, self.output_mapping)

    def compute_specific(self, keys, data):
        """Compute output of specific item, without changing output_keys."""
        output_mapping = self._output_keys_to_mapping(keys)
        order = self.dg.get_evaluation_order(
            selected_keys=self.get_selected_node_ids(keys)
        )
        return self._compute(data, order, output_mapping)

    def _compute(self, data, order, output_mapping):
        if self.unaccounted_keys:
            MSG = "These keys are still unaccounted for in the data pipeline: "
            MSG += ", ".join(self.unaccounted_keys)
            raise RuntimeError(MSG)
        intermediate = {}
        for node_id, edges, item in order:
            if isinstance(item, StaticItem):
                # Static item in data.
                # Just check that key is found.
                try:
                    data[item.key]
                    continue
                except KeyError:
                    raise KeyError(f"Expected key {item.key} in data!")
            # A dynamic item, which we should compute:
            args = [
                data[argkey] if argkey in data else intermediate[argkey]
                for argkey in item.next_takes()
            ]
            # This needs to be called BEFORE the dynamic item is called.
            provided_keys = item.next_provides()
            values = item(*args)  # Call the DynamicItem to produce output
            # If there is just one output value, wrap in a list so that
            # it can be zipped as well:
            if len(provided_keys) == 1:
                values = [values]
            intermediate.update(zip(provided_keys, values))
        for dynamic_item in self.dynamic_items:
            dynamic_item.reset()
        return {
            outkey: data[inkey] if inkey in data else intermediate[inkey]
            for outkey, inkey in output_mapping.items()
        }

    def get_selected_node_ids(self, selected_keys):
        """Translates selected keys to dependency graph keys."""
        return [self.key_to_node[key] for key in selected_keys]

    def __call__(self, data):
        return self.compute_outputs(data)

    def _prepare_run(self, data):
        self._exec_order = list(
            self.dg.get_evaluation_order(
                self.get_selected_node_ids(self.output_mapping.values())
            )
        )


DGNode = collections.namedtuple("DGNode", ["key", "edges", "data"])
# A node in DependencyGraph.


class DependencyGraph:
    """General-purpose dependency graph.

    Essentially a directed acyclic graph.
    Usually used to find an evaluation order for e.g. variable substitution
    The relation that an edge between A and B represents is:
    "A depends on B, i.e. B should be evaluated before A"

    Nodes can be added explicitly or they can be created implicitly
    while adding edges.
    Nodes have keys, which should be some hashable value that identifies
    the elements the graph represents in your use case. E.G. they can just
    be the variable name you want to substitute.
    However, if needed, more generally you can attach any data to a node
    (e.g. a path in your tree), and if so desired, a unique key can be
    created for you. You'll only need to know that key while adding edges
    to/from it.
    Implicit keys and explicit keys can also be mixed.
    """

    def __init__(self):
        self.digraph = []
        self.key2ind = {}
        # Guard for manual duplicates (but not implicitly added ones)
        self._manually_added_keys = []

    @staticmethod
    def get_unique_key():
        # Returns a unique hashable identifier
        return uuid.uuid4()

    def add_node(self, key=None, data=None):
        """Adds a node explicitly.

        Arguments
        ---------
        key : hashable, optional
            If not given, a key is created for you.
        data : Any, optional
            Any additional data you wish to attach to this node.

        Returns
        -------
        hashable
            The key that was used (either yours or generated).

        Raises
        ------
        ValueError
            If node with the given key has already been added explicitly
            (with this method, not "add_edge").
        """
        if key is None:
            key = self.get_unique_key()
        elif key in self._manually_added_keys:
            raise ValueError("Adding duplicate node: {key}".format(key=key))
        else:
            self._manually_added_keys.append(key)
        if key in self.key2ind:  # Implicitly added already; don't add again.
            ind = self.key2ind[key]
            node = self.digraph[ind]
            # All that this operation can do is add data:
            self.digraph[ind] = DGNode(node.key, node.edges, data)
            return key
        self.key2ind[key] = len(self.digraph)
        self.digraph.append(DGNode(key, [], data))
        return key

    def add_edge(self, from_key, to_key):
        """Adds an edge, and implicitly also creates nodes for keys which have
        not been seen before. This will not let you add data to your nodes.
        The relation encodes: "from_key depends on to_key"
        (to_key must be evaluated before from_key).

        Arguments
        ---------
        from_key : hashable
            The key which depends on.
        to_key : hashable
            The key which is depended on.

        Returns
        -------
        None
        """
        from_ind = self._get_ind_and_add_if_new(from_key)
        to_ind = self._get_ind_and_add_if_new(to_key)
        edges_list = self.digraph[from_ind].edges
        if to_ind not in edges_list:
            edges_list.append(to_ind)

    def _get_ind_and_add_if_new(self, key):
        # Used internally to implicitly add nodes for unseen keys
        if key not in self.key2ind:
            self.key2ind[key] = len(self.digraph)
            self.digraph.append(DGNode(key, [], None))
        return self.key2ind[key]

    def is_valid(self):
        """Checks if an evaluation order can be found.

        A dependency graph is evaluatable if there are no circular
        dependencies, i.e., the graph is acyclic.

        Returns
        -------
        bool
            Indicating if the graph is evaluatable.
        """
        return not self._find_first_cycle()

    def get_evaluation_order(self, selected_keys=None):
        """Finds one valid evaluation order.

        There can be many different valid
        orders.
        NOTE: Generates output one DGNode at a time. May generate DGNodes
        before it finds a circular dependency. If you really need to know
        whether an order can be found, check is_valid() first. However,
        the algorithm for finding cycles is essentially the same as the one
        used for finding an evaluation order, so for very large graphs...
        Ah well, but maybe then you should be using some other solution
        anyway.

        Arguments
        ---------
        selected_keys : list, None
            List of keys. If not None, only the selected keys are guaranteed
            in the evaluation order (along with the keys they depend on).

        Yields
        ------
        DGNode
            The added DGNodes in a valid evaluation order.
            See the DGNode namedtuple above.

        Raises
        ------
        CircularDependencyError
            If a circular dependency is found.
        """
        seen_ever = set()

        def toposort(root_ind, visited):
            nonlocal seen_ever
            here = visited + [root_ind]
            if root_ind in visited:
                raise CircularDependencyError(
                    "{cycle}".format(
                        cycle=" -> ".join(
                            str(self.digraph[i].key) for i in here
                        )
                    )
                )
            if root_ind in seen_ever:
                return  # Yield nothing
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                for ind in toposort(to_ind, visited=here):
                    yield ind
            yield root_ind

        if selected_keys is None:
            start_inds = range(len(self.digraph))
        else:
            start_inds = [self.key2ind[key] for key in selected_keys]

        for start_ind in start_inds:
            for ind in toposort(start_ind, []):
                yield self.digraph[ind]

    def _find_first_cycle(self):
        # Depth-first search based algorithm for finding cycles in the graph
        seen_ever = set()

        def cycle_dfs(root_ind, visited):
            nonlocal seen_ever
            print(root_ind, visited)
            here = visited + [root_ind]
            if root_ind in visited:
                return here
            if root_ind in seen_ever:
                return []
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                cycle = cycle_dfs(to_ind, here)
                if cycle:
                    return cycle
            return []

        for ind in range(len(self.digraph)):
            if ind not in seen_ever:
                cycle = cycle_dfs(ind, [])
                if cycle:
                    return cycle
        return []

    def __contains__(self, key):
        # Allows the syntax:
        # 'key' in dependency_graph
        return key in self.key2ind


class CircularDependencyError(ValueError):
    """
    An error caused by running into circular dependencies while searching for
    an evaluation order in a DependencyGraph.
    """

    pass


@dataclass
class StaticItem:
    """Data class that represents a static item.

    Static items are in-memory items so they don't need to be computed
    dynamically.
    """

    key: str


class ReproducibleRandomSampler(RandomSampler):
    """A modification of RandomSampler which always returns the same values.

    Also look at `torch.utils.data.RandomSampler`. This has mostly
    the same behaviour and arguments, except for adding 'seed' and 'epoch' and
    not supporting 'generator'.

    Note
    ----
    Call `set_epoch` before every epoch. Otherwise, the sampler will produce the
    same sequence of indices every epoch.

    Arguments
    ---------
    data_source : Dataset
        The data source to sample indices for.
    seed : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    epoch : int
        The epoch to start at.

    Example
    -------
    import torch
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.dataio.dataloader import SaveableDataLoader
    # An example "dataset"
    dataset = torch.arange(10).unsqueeze(1)
    # Create the random sampler:
    sampler = ReproducibleRandomSampler(dataset)
    dataloader = SaveableDataLoader(dataset, sampler = sampler,
    ...     num_workers = 3)
    # Setup the checkpointer.
    # Note that the sampler doesn't need to be saved itself.
    tmpdir = getfixture('tmpdir')
    checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
    # Iterate:
    subset = []
    for i, data_point in enumerate(dataloader):
    ...     # Say you save a checkpoint on the fourth batch:
    ...     if i == 3:
    ...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
    ...     # So let's save the numbers you would get if you continue
    ...     if i >= 4:
    ...         subset.append(data_point.item())
    # What if instead you had to restart the experiment?
    new_sampler = ReproducibleRandomSampler(dataset)
    new_dataloader = SaveableDataLoader(dataset, sampler = new_sampler,
    ...        num_workers = 3)
    new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
    _ = new_checkpointer.recover_if_possible()
    # You'll get the same random order again:
    new_subset = [data_point.item() for data_point in new_dataloader]
    assert subset == new_subset

    """

    def __init__(self, data_source, seed=563375142, epoch=0, **kwargs):
        if "generator" in kwargs:
            MSG = (
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            )
            raise ValueError(MSG)
        super().__init__(data_source, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        self.generator = torch.Generator()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self.epoch = epoch

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


def make_dataloader(dataset, looped_nominal_epoch=None, **loader_kwargs):
    """Makes a basic DataLoader with SpeechBrain defaults.

    For DynamicItemDatasets (which return dicts), use
    PaddedBatch as the default collate_fn.

    Shuffling gets implemented by ReproducibleRandomSampler.

    If the Dataset is not an IterableDataset, the DataLoader
    is a SaveableDataLoader.

    If the Dataset is a webdataset.dataset.Composable, set default
    batch_size = None.

    Can also loop over the underlying dataloader continuously,
    and stop iterations at nominal epoch lengths.

    Arguments
    ---------
    dataset : Dataset
        The dataset to make a DataLoader for.
    looped_nominal_epoch : None, int
        If an integer is given, loop the underlying DataLoader infinitely and
        set a nominal epoch length in batches (or whatever the DataLoader
        yields).
    **loader_kwargs : dict
        Keyword args to DataLoader, see PyTorch DataLoader for
        options.

    Returns
    -------
    DataLoader
        If looped_nominal_epoch is None
    LoopedLoader
        If looped_nominal_epoch is not None
    """
    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]
    # With WDS it is recommended to do batching in the dataset itself,
    # which requires batch_size = None in the DataLoader
    if (
        WDS_AVAILABLE
        and isinstance(dataset, wds.dataset.Composable)
        and "batch_size" not in loader_kwargs
    ):
        loader_kwargs["batch_size"] = None
    # Create the loader
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        dataloader = SaveableDataLoader(dataset, **loader_kwargs)
    if looped_nominal_epoch is not None:
        dataloader = LoopedLoader(dataloader, looped_nominal_epoch)
    return dataloader


PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])


def batch_pad_right(tensors: list, mode="constant", value=0):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        any(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


class PaddedBatch:
    """Collate_fn when examples are dicts and have variable-length sequences.

    Different elements in the examples get matched by key.
    All numpy tensors get converted to Torch (PyTorch default_convert)
    Then, by default, all torch.Tensor valued elements get padded and support
    collective pin_memory() and to() calls.
    Regular Python data types are just collected in a list.

    Arguments
    ---------
    examples : list
        List of example dicts, as produced by Dataloader.
    padded_keys : list, None
        (Optional) List of keys to pad on. If None, pad all torch.Tensors
    device_prep_keys : list, None
        (Optional) Only these keys participate in collective memory pinning and moving with
        to().
        If None, defaults to all items with torch.Tensor values.
    padding_func : callable, optional
        Called with a list of tensors to be padded together. Needs to return
        two tensors: the padded data, and another tensor for the data lengths.
    padding_kwargs : dict
        (Optional) Extra kwargs to pass to padding_func. E.G. mode, value
    apply_default_convert : bool
        Whether to apply PyTorch default_convert (numpy to torch recursively,
        etc.) on all data. Default:True, usually does the right thing.
    nonpadded_stack : bool
        Whether to apply PyTorch-default_collate-like stacking on values that
        didn't get padded. This stacks if it can, but doesn't error out if it
        cannot. Default:True, usually does the right thing.

    Example
    -------
    batch = PaddedBatch([
    ...     {"id": "ex1", "foo": torch.Tensor([1.])},
    ...     {"id": "ex2", "foo": torch.Tensor([2., 1.])}])
    # Attribute or key-based access:
    batch.id
    ['ex1', 'ex2']
    batch["id"]
    ['ex1', 'ex2']
    # torch.Tensors get padded
    type(batch.foo)
    <class 'speechbrain.dataio.batch.PaddedData'>
    batch.foo.data
    tensor([[1., 0.],
            [2., 1.]])
    batch.foo.lengths
    tensor([0.5000, 1.0000])
    # Batch supports collective operations:
    _ = batch.to(dtype=torch.half)
    batch.foo.data
    tensor([[1., 0.],
            [2., 1.]], dtype=torch.float16)
    batch.foo.lengths
    tensor([0.5000, 1.0000], dtype=torch.float16)
    # Numpy tensors get converted to torch and padded as well:
    import numpy as np
    batch = PaddedBatch([
    ...     {"wav": np.asarray([1,2,3,4])},
    ...     {"wav": np.asarray([1,2,3])}])
    batch.wav  # +ELLIPSIS
    PaddedData(data=tensor([[1, 2,...
    # Basic stacking collation deals with non padded data:
    batch = PaddedBatch([
    ...     {"spk_id": torch.tensor([1]), "wav": torch.tensor([.1,.0,.3])},
    ...     {"spk_id": torch.tensor([2]), "wav": torch.tensor([.2,.3,-.1])}],
    ...     padded_keys=["wav"])
    batch.spk_id
    tensor([[1],
            [2]])
    # And some data is left alone:
    batch = PaddedBatch([
    ...     {"text": ["Hello"]},
    ...     {"text": ["How", "are", "you?"]}])
    batch.text
    [['Hello'], ['How', 'are', 'you?']]

    """

    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], torch.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.

        Example
        -------
        batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        ids, vals = batch
        ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Fetch an item by its position in the batch."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        return self.__length


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers.

    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def mod_default_collate(batch):
    r"""Makes a tensor from list of batch values.

    Note that this doesn't need to zip(*) values together
    as PaddedBatch connects them already (by key).

    Here the idea is not to error out.

    This is modified from:
    https://github.com/pytorch/pytorch/blob/c0deb231db76dbea8a9d326401417f7d1ce96ed5/torch/utils/data/_utils/collate.py#L42
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:  # Unequal size:
            return batch
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        try:
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    return batch
                return mod_default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        except RuntimeError:  # Unequal size
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    else:
        return batch


# @register_checkpoint_hooks
class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    SpeechBrain's Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speechbrain_iterator = iterator
        return iterator

    # @mark_as_saver
    def _speechbrain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = self._speechbrain_iterator._num_yielded
        with open(path, "w") as fo:
            fo.write(str(to_save))

    # @mark_as_loader
    def _speechbrain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._speechbrain_recovery_skip_to = int(saved)


# @register_checkpoint_hooks
class LoopedLoader:
    """Loops an underlying iterable indefinitely, with nominal epoch lengths

    This is useful for working with IterableDatasets, and particularly
    webdataset-style loading. We recommend using ``.repeat()`` on the
    webdataset IterableDataset instance, so that the underlying dataloader
    naturally continues for ever.

    Arguments
    ---------
    loader : iterable
        A DataLoader or other iterable that is looped repeatedly.
    epoch_length : int
        The length of the nominal epoch. After this many steps, raises
        StopIteration
    """

    def __init__(self, loader, epoch_length, batchsize_fn=None):
        self.loader = loader
        self.iterator = None
        self.epoch_length = epoch_length
        self.step = 0  # Step in epoch
        self.total_steps = 0  # Total steps ever
        self.total_samples = 0  # Total samples seen on this process
        if batchsize_fn is None:
            self.batchsize_fn = BatchsizeGuesser()

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.step < self.epoch_length:
            self.step += 1
            self.total_steps += 1
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            self.total_samples += self.batchsize_fn(batch)
            return batch
        else:
            self.step = 0
            raise StopIteration

    def __len__(self):
        return self.epoch_length

    # @mark_as_saver
    def save(self, path):
        with open(path, "w") as fo:
            print(self.step, file=fo)
            print(self.total_steps, file=fo)
            print(self.total_samples, file=fo)

    # @mark_as_loader
    def load(self, path, end_of_epoch=True, device=None):
        del device  # Unused here
        with open(path) as fi:
            self.step = int(fi.readline().strip())
            self.total_steps = int(fi.readline().strip())
            self.total_samples = int(fi.readline().strip())
            if not end_of_epoch and self.step == 0 and self.total_steps > 0:
                # Step has been set to 0 at the end of iteration,
                # so return it to epoch_length, so that first iteration
                # of this will immediately raise StopIteration.
                # Basically, this can happen when e.g. the main training
                # loop has already finished but there is a checkpoint in the
                # middle of validation.
                self.step = self.epoch_length


class BatchsizeGuesser:
    """Try to figure out the batchsize, but never error out

    If this cannot figure out anything else, will fallback to guessing 1

    Example
    -------
    guesser = BatchsizeGuesser()
    # Works with simple tensors:
    guesser(torch.randn((2,3)))
    2
    # Works with sequences of tensors:
    guesser((torch.randn((2,3)), torch.randint(high=5, size=(2,))))
    2
    # Works with PaddedBatch:
    guesser(PaddedBatch([{"wav": [1.,2.,3.]}, {"wav": [4.,5.,6.]}]))
    2
    guesser("Even weird non-batches have a fallback")
    1

    """

    def __init__(self):
        self.method = None

    def __call__(self, batch):
        try:
            return self.method(batch)
        except:  # noqa: E722
            return self.find_suitable_method(batch)

    def find_suitable_method(self, batch):
        """Try the different methods and note which worked"""
        try:
            bs = self.attr_based(batch)
            self.method = self.attr_based
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.torch_tensor_bs(batch)
            self.method = self.torch_tensor_bs
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.len_of_first(batch)
            self.method = self.len_of_first
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.len_of_iter_first(batch)
            self.method = self.len_of_iter_first
            return bs
        except:  # noqa: E722
            pass
        # Last ditch fallback:
        bs = self.fallback(batch)
        self.method = self.fallback(batch)
        return bs

    def attr_based(self, batch):
        return batch.batchsize

    def torch_tensor_bs(self, batch):
        return batch.shape[0]

    def len_of_first(self, batch):
        return len(batch[0])

    def len_of_iter_first(self, batch):
        return len(next(iter(batch)))

    def fallback(self, batch):
        return 1


def ddp_barrier():
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def if_main_process():
    """Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.
    """
    if "RANK" in os.environ:
        if os.environ["RANK"] == "":
            return False
        else:
            if int(os.environ["RANK"]) == 0:
                return True
            return False
    return True


"""
Convert object based augmentations into functions
"""

f_augment_wavedrop = TimeDomainSpecAugment(sample_rate=16000, speeds=[100])
f_augment_speed = TimeDomainSpecAugment(sample_rate=16000, speeds=[95, 100, 105])
f_add_rev = EnvCorrupt(openrir_folder=OPENRIR_FOLDER, openrir_max_noise_len=3.0, reverb_prob=1.0, noise_prob=0.0, noise_snr_low=0, noise_snr_high=15, rir_scale_factor=1.0)
f_add_noise = EnvCorrupt(openrir_folder=OPENRIR_FOLDER, openrir_max_noise_len=3.0, reverb_prob=0.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, rir_scale_factor=1.0)
f_add_rev_noise = EnvCorrupt(openrir_folder=OPENRIR_FOLDER, openrir_max_noise_len=3.0, reverb_prob=1.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, rir_scale_factor=1.0)


def augment_wavedrop(audio, lengths, sample_rate=16000):
    return f_augment_wavedrop(audio, lengths)


def augment_speed(audio, lengths, sample_rate=16000):
    return f_augment_speed(audio, lengths)


def add_rev(audio, lengths, sample_rate=16000):
    return f_add_rev(audio, lengths)


def add_noise(audio, lengths, sample_rate=16000):
    return f_add_noise(audio, lengths)


def add_rev_noise(audio, lengths, sample_rate=16000):
    return f_add_rev_noise(audio, lengths)
