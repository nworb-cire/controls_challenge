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

    def __init__(self, data, sequence_length):
        assert data.ndim == 3
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.CONTEXT_SIZE - self.sequence_length)

    def __getitem__(self, idx):
        seq_idx = idx // (self.data.shape[1] - self.CONTEXT_SIZE - self.sequence_length)
        start_idx = idx % (self.data.shape[1] - self.CONTEXT_SIZE - self.sequence_length)

        return self.data[seq_idx, start_idx:start_idx + self.CONTEXT_SIZE + self.sequence_length, :]


class DataModule(pl.LightningDataModule):
    x_cols = [
        "steerCommand",
        "roll",
        "vEgo",
        "aEgo",
    ]
    y_col = "targetLateralAcceleration"

    def __init__(self, sequence_length: int = 100):
        super().__init__()
        self.sequence_length = sequence_length

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
        self.files = glob(f"{DATASET_PATH}/*.csv")
        for file in self.files:
            df = pd.read_csv(file)
            df = df[self.x_cols + [self.y_col]]
            df["roll"] = np.sin(df["roll"]) * 9.81
            # Allow 20 rows with null values
            not_na_rows = df[df["steerCommand"].notna()].index.max()
            df = df.iloc[:not_na_rows + self.sequence_length + 1]
            # add batch dimension
            val = df.values[np.newaxis]
            segments.append(val)

        # Concatenate: (n_sequences, sequence_length, n_features)
        data = np.concatenate(segments, axis=0)

        self.train = LatAccelDataset(data, sequence_length=self.sequence_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=1024, shuffle=True)

    def val_dataloader(self):
        return self.files[:100]
