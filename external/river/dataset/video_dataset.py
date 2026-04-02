import albumentations
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from .h5 import HDF5Dataset


class Aug(nn.Module):
    def __init__(self, b: float, c: float, s: float, h: float):
        super(Aug, self).__init__()

        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        im = F.adjust_brightness(im, brightness_factor=1 + self.b)
        im = F.adjust_contrast(im, contrast_factor=1 + self.c)
        im = F.adjust_saturation(im, saturation_factor=1 + self.s)
        im = F.adjust_hue(im, hue_factor=self.h)

        return im


class RandomConsistentAugFactory(nn.Module):
    def __init__(self, aug: bool = True):
        super(RandomConsistentAugFactory, self).__init__()

        self.aug = aug

    def forward(self):
        if self.aug:
            b = (torch.rand(1).item() - 0.5) / 5
            c = (torch.rand(1).item() - 0.5) / 5
            s = (torch.rand(1).item() - 0.5) / 5
            h = (torch.rand(1).item() - 0.5) / 2
            aug = Aug(b, c, s, h)

            return aug

        else:
            return T.Lambda(lambda x: x)


# class VideoDataset(Dataset):

#     def __init__(
#             self,
#             data_path,
#             input_size: int,
#             crop_size: int,
#             frames_per_sample=5,
#             skip_frames=0,
#             random_time=True,
#             random_horizontal_flip=True,
#             aug=False,
#             albumentations=False,
#             total_videos=-1):

#         self.data_path = data_path
#         self.frames_per_sample = frames_per_sample
#         self.random_time = random_time
#         self.skip_frames = skip_frames
#         self.random_horizontal_flip = random_horizontal_flip
#         self.total_videos = total_videos

#         self.albumentations = albumentations

#         self.input_size = input_size
#         self.crop_size = crop_size

#         self.aug = RandomConsistentAugFactory(aug)

#         # Read h5 files as dataset
#         self.videos_ds = HDF5Dataset(self.data_path)

#         print(f"Dataset length: {self.__len__()}")

#     def __len__(self):
#         return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

#     def max_index(self):
#         return len(self.videos_ds)

#     def __getitem__(self, index, time_idx=0):
#         video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
#         shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

#         # Setup augmentations
#         flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
#         if self.albumentations:
#             tr = albumentations.Compose([
#                 albumentations.SmallestMaxSize(max_size=self.input_size),
#                 albumentations.CenterCrop(height=self.crop_size, width=self.crop_size),
#                 albumentations.HorizontalFlip(p=flip_p)
#             ])
#         else:
#             tr = T.Compose([
#                 T.Resize(size=self.input_size, antialias=True),
#                 T.CenterCrop(size=self.crop_size),
#                 T.RandomHorizontalFlip(p=flip_p)
#             ])
#         color_tr = self.aug()

#         prefinals = []
#         #print("self.random_time", self.random_time)
#         with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
#             video_len = f['len'][str(idx_in_shard)][()]
#             num_frames = (self.skip_frames + 1) * (self.frames_per_sample - 1) + 1
#             #print("video_len",video_len, "num_frames",num_frames)
#             assert video_len >= num_frames, "The video is shorter than the desired sample size"
#             if self.random_time:
#                 time_idx = np.random.choice(video_len - num_frames)
#             assert time_idx < video_len, "Time index out of video boundary"
#             for i in range(time_idx, min(time_idx + num_frames, video_len), self.skip_frames + 1):
#                 img = f[str(idx_in_shard)][str(i)][()]
#                 if self.albumentations:
#                     arr = tr(image=img)["image"]
#                 else:
#                     arr = img
#                 prefinals.append(torch.Tensor(arr).to(torch.uint8))

#         data = torch.stack(prefinals)
#         if not self.albumentations:
#             data = tr(data.permute(0, 3, 1, 2))
#         else:
#             data = data.permute(0, 3, 1, 2)
#         data = color_tr(data).to(torch.float32) / 127.5 - 1.0

#         return data

class VideoDataset(Dataset):
    def __init__(
            self,
            data_path,
            input_size: int,
            crop_size: int,
            frames_per_sample=5,
            skip_frames=0,
            random_time=True,
            random_horizontal_flip=False,
            aug=False,
            albumentations=True,
            total_videos=-1,
            return_raw_frames=False):

        self.data_path = data_path
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.skip_frames = skip_frames
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos
        self.albumentations = albumentations
        self.input_size = input_size
        self.crop_size = crop_size
        self.return_raw_frames = return_raw_frames

        self.aug = RandomConsistentAugFactory(aug)

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        # Setup augmentations
        flip_p = float(np.random.randint(2) == 0) if self.random_horizontal_flip else 0.0

        if self.albumentations:
            tr = albumentations.Compose([
                albumentations.SmallestMaxSize(max_size=self.input_size),
                albumentations.CenterCrop(height=self.crop_size, width=self.crop_size),
                albumentations.HorizontalFlip(p=flip_p),
            ])
        else:
            tr = T.Compose([
                T.Resize(size=self.input_size, antialias=True),
                T.CenterCrop(size=self.crop_size),
                T.RandomHorizontalFlip(p=flip_p),
            ])

        color_tr = self.aug()

        prefinals = []
        raw_frames = [] if self.return_raw_frames else None

        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = f["len"][str(idx_in_shard)][()]
            num_frames = (self.skip_frames + 1) * (self.frames_per_sample - 1) + 1

            assert video_len >= num_frames, "The video is shorter than the desired sample size"

            if self.random_time:
                max_start = video_len - num_frames
                time_idx = np.random.randint(0, max_start + 1)
            else:
                time_idx = 0

            assert time_idx < video_len, "Time index out of video boundary"

            # Sampled clip
            for i in range(time_idx, min(time_idx + num_frames, video_len), self.skip_frames + 1):
                img = f[str(idx_in_shard)][str(i)][()]   # [H, W, C], uint8

                if self.albumentations:
                    arr = tr(image=img)["image"]         # still [H, W, C]
                    frame = torch.from_numpy(arr)
                else:
                    frame = torch.from_numpy(img)

                prefinals.append(frame.to(torch.uint8))

            # Full sequence, only if requested
            if self.return_raw_frames:
                for i in range(video_len):
                    img = f[str(idx_in_shard)][str(i)][()]
                    raw_frames.append(torch.from_numpy(img).to(torch.uint8))

        # Stack sampled frames: [T, H, W, C]
        data = torch.stack(prefinals)

        # Apply torchvision transforms if not using albumentations
        if not self.albumentations:
            data = tr(data.permute(0, 3, 1, 2))   # [T, C, H, W]
        else:
            data = data.permute(0, 3, 1, 2)       # [T, C, H, W]

        # Convert to float32 in [0,1]
        data = data.to(torch.float32) / 255.0

        # Optional color augmentation
        data = color_tr(data)

        if self.return_raw_frames:
            raw_frames = torch.stack(raw_frames)   # [T_full, H, W, C], uint8
            return data, raw_frames

        return data
        

class FullSequenceEvalDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_size: int,
        crop_size: int,
        skip_frames=0,
        use_albumentations=True,
        total_videos=-1,
    ):
        self.data_path = data_path
        self.skip_frames = skip_frames
        self.use_albumentations = use_albumentations
        self.input_size = input_size
        self.crop_size = crop_size
        self.total_videos = total_videos

        self.videos_ds = HDF5Dataset(self.data_path)

        if self.use_albumentations:
            self.tr = albumentations.Compose([
                albumentations.SmallestMaxSize(max_size=self.input_size),
                albumentations.CenterCrop(height=self.crop_size, width=self.crop_size),
            ])
        else:
            self.tr = T.Compose([
                T.Resize(size=self.input_size, antialias=True),
                T.CenterCrop(size=self.crop_size),
            ])

        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index):
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        frames = []

        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = int(f["len"][str(idx_in_shard)][()])

            # load full sequence from frame 0
            for i in range(0, video_len, self.skip_frames + 1):
                img = f[str(idx_in_shard)][str(i)][()]   # [H, W, C], uint8

                if self.use_albumentations:
                    arr = self.tr(image=img)["image"]   # still [H, W, C]
                    frame = torch.from_numpy(arr)
                else:
                    frame = torch.from_numpy(img)

                frames.append(frame.to(torch.uint8))

        # [T, H, W, C]
        data = torch.stack(frames)

        # convert to [T, C, H, W]
        if not self.use_albumentations:
            data = self.tr(data.permute(0, 3, 1, 2))
        else:
            data = data.permute(0, 3, 1, 2)

        # float in [0,1]
        data = data.to(torch.float32) / 255.0

        seq_len = data.shape[0]

        return data, seq_len
        
def pad_collate_fullseq(batch):
    """
    batch: list of (data, seq_len)
      data: [T_i, 3, H, W]
      seq_len: int

    returns:
      padded_data: [B, T_max, 3, H, W]
      seq_len: [B]
    """
    data_list, seq_lens = zip(*batch)

    max_len = max(seq_lens)
    B = len(data_list)
    C, H, W = data_list[0].shape[1:]

    padded = torch.zeros(B, max_len, C, H, W, dtype=data_list[0].dtype)

    for i, x in enumerate(data_list):
        T_i = x.shape[0]
        padded[i, :T_i] = x

    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    return padded, seq_lens