import os
from typing import Tuple

import torch
from torch import Tensor
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision import transforms

from data import get_clip_timepoints, waveform2melspec

def load_and_transform_single_audio_data(
    audio_path,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    waveform = waveform.repeat(1, 4)

    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    return all_clips

def _load_waveform(
    root: str,
    filename: str,
    exp_sample_rate: int,
):
    path = os.path.join(root, filename)
    waveform = load_and_transform_single_audio_data(path, sample_rate=exp_sample_rate)

    return waveform

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./.datasets/speechcommands", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        
        self.labels = ['backward',
                    'bed', 'bird', 'cat', 'dog', 'down',
                    'eight', 'five', 'follow', 'forward', 'four',
                    'go', 'happy', 'house', 'learn', 'left',
                    'marvin', 'nine', 'no', 'off', 'on',
                    'one', 'right', 'seven', 'sheila', 'six',
                    'stop', 'three', 'tree', 'two', 'up',
                    'visual', 'wow', 'yes', 'zero']
    
    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            str:
                Label
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + (self.label_to_index(metadata[2]),)
