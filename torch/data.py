import logging
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

from tinyphysics import DATASET_PATH, DATASET_URL, CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX

FUTURE_PLAN_LENGTH = 20


class LatAccelDataset(Dataset):
    def __init__(self, data):
        assert data.ndim == 3
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :]


class DataModule(pl.LightningDataModule):
    x_cols = [
        "steerCommand",
        "roll",
        "vEgo",
        "aEgo",
    ]
    y_col = "targetLateralAcceleration"

    def __init__(self):
        super().__init__()

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
            if len(df) < COST_END_IDX + FUTURE_PLAN_LENGTH:
                logging.warning(f"Skipping {file} due to insufficient length")
                continue
            df = df[self.x_cols + [self.y_col]]
            df["roll"] = np.sin(df["roll"]) * 9.81
            df = df.iloc[CONTROL_START_IDX - CONTEXT_LENGTH:COST_END_IDX + FUTURE_PLAN_LENGTH]
            # add batch dimension
            val = df.values[np.newaxis]
            segments.append(val)

        # Concatenate: (n_sequences, sequence_length, n_features)
        data = np.concatenate(segments, axis=0)

        self.train = LatAccelDataset(data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=4096, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return self.files[:100]
