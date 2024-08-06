import os
import urllib.request
import zipfile
from io import BytesIO

import pytorch_lightning as pl

from tinyphysics import DATASET_PATH, DATASET_URL


# class


class DataModule(pl.LightningDataModule):
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

    def setup(self, stage):
        pass
