"""TokenDataset: reads token cache (memmap fp16) + tabular features from snapshot."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, image_tokens, text_tokens, text_mask,
                 tabular_continuous, genre_idx, targets, video_ids):
        self.image_tokens = image_tokens   # memmap (N, 50, 768) fp16
        self.text_tokens = text_tokens     # memmap (N, 32, 512) fp16
        self.text_mask = text_mask         # memmap (N, 32) bool
        self.tabular = tabular_continuous  # (N, n_cont) fp32
        self.genre_idx = genre_idx         # (N,) int64
        self.targets = targets             # (N,) fp32
        self.video_ids = video_ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {
            "image_tokens": torch.from_numpy(np.array(self.image_tokens[i], dtype=np.float32)),
            "text_tokens": torch.from_numpy(np.array(self.text_tokens[i], dtype=np.float32)),
            "text_mask": torch.from_numpy(np.array(self.text_mask[i])),
            "tabular": torch.from_numpy(np.array(self.tabular[i], dtype=np.float32)),
            "genre_idx": int(self.genre_idx[i]),
            "target": np.float32(self.targets[i]),
            "video_id": self.video_ids[i],
        }