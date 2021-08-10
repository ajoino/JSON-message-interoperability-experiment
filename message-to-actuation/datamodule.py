from functools import partial
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from dataset import MessageDataset

class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.setpoint_transform = transforms.Compose([
            torch.Tensor,
            lambda x: (x - 296) / 10,
        ])
        self.actuation_transform = transforms.Compose([
            torch.Tensor,
            lambda x: (x - 12) / 333
        ])
        self.room_name_transform = transforms.Compose([
            torch.LongTensor,
            partial(nn.functional.one_hot, num_classes=6)
        ])

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage is None:
            self.message_data_train, self.message_data_val = random_split(
                    Subset(md := MessageDataset(
                        self.data_dir,
                        room_transform=self.room_name_transform,
                        setpoint_transform=self.setpoint_transform,
                        actuation_transform=self.actuation_transform,
                    ), [i for i in range(60000)])
                    , [55000, 5000]
            )

        if stage == 'test' or stage is None:
            self.message_data_test = Subset(
                    md := MessageDataset(
                    self.data_dir,
                    room_transform=self.room_name_transform,
                    setpoint_transform=self.setpoint_transform,
                    actuation_transform=self.actuation_transform
                    ), [i for i in range(len(md) - 10000, len(md))]
            )


    def train_dataloader(self):
        return DataLoader(self.message_data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.message_data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.message_data_test, batch_size=200)