import os
import logging

import torch
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.utils import verify_str_arg
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo

from data import get_clip_timepoints, SpatialCrop

class myKinetics400(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train"
    ) -> None:
        self.root = root
        self.split = verify_str_arg(split, arg="split", valid_values=["train", "val", "test"])
        self.split_folder = os.path.join(root, split)
        extensions = ("avi", "mp4")
        self.video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(224),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        super().__init__(self.root)
        self.classes, class_to_idx = find_classes(self.split_folder)
        self.samples = make_dataset(self.split_folder, class_to_idx, extensions, is_valid_file=None)
    
    def load_and_transform_video_data(
        self,
        video_path,
        clip_duration=2,
        clips_per_video=5,
        sample_rate=16000,
    ):
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [self.video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)
        all_video = torch.stack(all_video, dim=0)
        return all_video

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        try:
            video = self.load_and_transform_video_data(video_path)
        except RuntimeError as e:
            logging.warning(f'{e}')
            logging.warning(f'RuntimeError in loading video - {video_path}')
            video = torch.zeros((15, 3, 2, 224, 224))
        if video.shape != (15, 3, 2, 224, 224):
            logging.warning(f'video.shape != (15, 3, 2, 224, 224) - {video_path}')
            video = torch.zeros((15, 3, 2, 224, 224))
        return video, label
