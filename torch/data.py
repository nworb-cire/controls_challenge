import os
import urllib.request
import zipfile
from glob import glob
from io import BytesIO

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from tinyphysics import DATASET_PATH, DATASET_URL


class LatAccelDataset(Dataset):
    CONTEXT_SIZE = 20
    SEQUENCE_LENGTH = 100

    def __init__(self, data):
        assert data.ndim == 3
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0] * (self.SEQUENCE_LENGTH - self.CONTEXT_SIZE * 2)

    def __getitem__(self, idx):
        seq_idx = idx // (self.SEQUENCE_LENGTH - self.CONTEXT_SIZE * 2)
        start_idx = idx % (self.SEQUENCE_LENGTH - self.CONTEXT_SIZE * 2)

        return self.data[seq_idx, start_idx:start_idx + self.CONTEXT_SIZE * 2, :]


class DataModule(pl.LightningDataModule):
    x_cols = [
        "steerCommand",
        "roll",
        "vEgo",
        "aEgo",
    ]
    y_col = "targetLateralAcceleration"

    def prepare_data(self) -> None:
        if not DATASET_PATH.exists():
            print("Downloading dataset (0.6G)...")
            DATASET_PATH.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(DATASET_URL) as resp:
                with zipfile.ZipFile(BytesIO(resp.read())) as z:
                    for member in z.namelist():
                        if not member.endswith('/'):
                            with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:
                                dest.write(src.read())

    def setup(self, stage: str = None):
        segments = []
        for file in glob(f"{DATASET_PATH}/*.csv"):
            df = pd.read_csv(file)
            df = df[self.x_cols + [self.y_col]]
            df["roll"] = np.sin(df["roll"]) * 9.81
            df = df.dropna()
            # add batch dimension
            val = df.values[np.newaxis]
            segments.append(val)

        # Concatenate: (n_sequences, sequence_length, n_features)
        data = np.concatenate(segments, axis=0)

        train_size = int(data.shape[0] * 0.9)
        print(f"Train size: {train_size:,}, Val size: {data.shape[0] - train_size:,}")
        self.train = LatAccelDataset(data[:train_size, :, :])
        self.val = LatAccelDataset(data[train_size:, :, :])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=256, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=512)
