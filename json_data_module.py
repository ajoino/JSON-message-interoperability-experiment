from typing import Union
from pathlib import Path
import json
from itertools import groupby
from collections import defaultdict
import string
from numbers import Number

from more_itertools import sort_together

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from json_parser import JSONParseTree


ALL_CHARACTERS = chr(3) + string.printable
NUM_CHARACTERS = len(ALL_CHARACTERS)
JSON_TYPES = ('___stop___',
              '___object___', '___array___',
              '___string___', '___number___',
              '___bool___', '___null___')

JSON_PRIMITIVES = JSON_TYPES[3:]

NUM_JSON_TYPES = len(JSON_TYPES)


class LeafDataToTensor:
    """Convert leaf data to tensors"""
    all_characters = string.printable

    def __call__(self, sample):
        tensor_sample = {identifier: {
            'type': data['type'],
            'leaf_data': self._transform(data['leaf_data']),
            'parse_tree': data['parse_tree']
        }
            for identifier, data in sample.items()}

        return tensor_sample

    def _transform(self, value):

        if isinstance(value, Number) or isinstance(value, bool):
            data = torch.Tensor([[value]])
        elif isinstance(value, str):
            data = torch.LongTensor(
                    [ALL_CHARACTERS.index(char) for char in value]
            )
        else:
            data = torch.zeros(1, 1)

        return data


class SimpleDataset(Dataset):
    """Simple dataset for json data"""

    def __init__(self, data_file, transform=None):

        with open(data_file, 'r') as json_file:
            self.jsons = json.load(json_file)
        self.transform = transform

    def __len__(self):
        return len(self.jsons)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Index has to be a single integer')

        tree = JSONParseTree.parse_start('___root___', self.jsons[idx])

        sample = {identifier: {'type': datum.type, 'leaf_data': datum.data, 'parse_tree': tree}
                  for identifier, datum in tree.leaf_data()}

        if self.transform:
            sample = self.transform(sample)

        return sample


class JSONDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: Union[str, Path] = './'):
        super().__init__()

        self.data_dir = Path(data_dir)

    def setup(self, stage=None):
        self.json_data = SimpleDataset(self.data_dir, transform=LeafDataToTensor())

    def train_dataloader(self):
        return DataLoader(self.json_data, collate_fn=lambda batch: batch, batch_size=4)


if __name__ == '__main__':
    data = SimpleDataset('some_json.json')
    data_module = JSONDataModule('some_json.json'); data_module.setup()
    print(data[0])
    for i, batch in enumerate(data_module.train_dataloader()):
        if i != 0:
            continue
        print(len(batch))
