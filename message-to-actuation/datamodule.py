from functools import partial
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

#import ujson as json
import json

from dataset import MessageDataset

def list_collator(batch, decode_json=False):
    collated = default_collate(batch)
    if decode_json:
        collated[0] = [json.loads(message) for message in collated[0]]
    return collated


class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32, decode_json: bool = False, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.decode_json = decode_json
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
        ])

    def setup(self, stage: Optional[str] = None):

        if stage in {'fit', 'predict'} or stage is None:
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
        return DataLoader(
                self.message_data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=partial(list_collator, decode_json=self.decode_json),
        )

    def val_dataloader(self):
        return DataLoader(
                self.message_data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=partial(list_collator, decode_json=self.decode_json),
        )

    def test_dataloader(self):
        return DataLoader(
                self.message_data_test,
                batch_size=200,
                num_workers=self.num_workers,
                collate_fn=partial(list_collator, decode_json=self.decode_json),
        )

    def predict_dataloader(self):
        return DataLoader(
                self.message_data_val,
                batch_size=200,
                num_workers=self.num_workers,
                collate_fn=partial(list_collator, decode_json=self.decode_json),
        )
