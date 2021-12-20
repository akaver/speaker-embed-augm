import augly.audio as audaugs
import torch

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
    aug_audio, sr = audaugs.clicks(audio.cpu().numpy(), sample_rate=sample_rate, seconds_between_clicks=0.5, snr_level_db=10.0)
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
