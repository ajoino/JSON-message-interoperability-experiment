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


def collate_tree_batches(batch):
    max_sequence_lengths = {identifier: max([len(sample_data['leaf_data'])])
                            for sample in batch
                            for identifier, sample_data in sample.items()}
    tensor_dict = defaultdict(list)
    index_dict = defaultdict(list)

    for sample in batch:
        for identifier, sample_data in sample.items():
            tensor_dict[identifier].append(sample_data['leaf_data'])
            index_dict[identifier].append(sample_data['sample_index'])

    collated_samples = {}
    for (identifier, index) in index_dict.items():
        tensors = tensor_dict[identifier]
        # TODO: isinstance check is not enough, to perfectly separate cases we need type information from the parse tree
        tensors = pad_sequence(tensors, padding_value=0) if isinstance(tensors[0], torch.LongTensor) else torch.cat(tensors, dim=0)
        index = torch.cat(index, dim=0)
        collated_samples[identifier] = {'sample_index': index, 'leaf_data': tensors}


    return collated_samples


class LeafDataToTensor:
    """Convert leaf data to tensors"""
    all_characters = string.printable

    def __call__(self, sample):
        tensor_sample = {identifier: {'sample_index': torch.LongTensor([[data['sample_index']]]), 'leaf_data': self._transform(data['leaf_data'])}
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

        sample = {identifier: {'sample_index': idx, 'leaf_data': datum} for identifier, datum in tree.leaf_data()}

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
        return DataLoader(self.json_data, collate_fn=collate_tree_batches, batch_size=4)


if __name__ == '__main__':
    from pprint import pprint
    jsons = JSONDataModule('some_json.json')
    jsons.setup()

    for batch in jsons.train_dataloader():
        pprint(batch)




