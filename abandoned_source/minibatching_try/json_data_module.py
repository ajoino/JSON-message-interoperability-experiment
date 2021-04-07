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


def collate_tree_batches(batch):
    max_sequence_lengths = {identifier: max([len(sample_data['leaf_data'])])
                            for sample in batch
                            for identifier, sample_data in sample.items()}
    tensor_dict = defaultdict(list)
    tree_dict = defaultdict(list)
    type_dict = defaultdict(list)
    tree_index_dict = defaultdict(list)

    unique_trees = list({value['parse_tree'] for data in batch for identifier, value in data.items()})

    for sample in batch:
        for identifier, sample_data in sample.items():
            tensor_dict[identifier].append(sample_data['leaf_data'])
            tree_dict[identifier].append(unique_trees)
            type_dict[identifier].append(sample_data['type'])
            tree_index_dict[identifier].append(unique_trees.index(sample_data['parse_tree']))

    collated_samples = {}
    for identifier in tensor_dict.keys():
        trees = tree_dict[identifier]
        tree_index = tree_index_dict[identifier]
        # TODO: isinstance check is not enough, to perfectly separate cases we need type information from the parse tree
        types = type_dict[identifier]
        tensors = tensor_dict[identifier]  # pad_sequence(tensors, padding_value=0) if isinstance(tensors[0], torch.LongTensor) else torch.cat(tensors, dim=0)
        type_masks = {type: torch.BoolTensor([tp == type for tp in types]) for type in types}
        masked_tensors = {type: cat_or_pad([
            tensor for tensor, m in zip(tensors, mask) if m
        ], type=type)
            for type, mask in type_masks.items()
        }
        collated_samples[identifier] = {
            'type': types,
            'leaf_data': masked_tensors,
            'parse_trees': trees,
            'tree_index': tree_index,
        }

    return collated_samples


def cat_or_pad(tensors, type):
    if type == '___string___':
        return pad_sequence(tensors, padding_value=0)

    return torch.cat(tensors, dim=0)


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
        return DataLoader(self.json_data, collate_fn=collate_tree_batches, batch_size=4)


if __name__ == '__main__':
    from pprint import pprint

    jsons = JSONDataModule('../../some_json.json')
    jsons.setup()

    for batch in jsons.train_dataloader():
        pprint(batch)
