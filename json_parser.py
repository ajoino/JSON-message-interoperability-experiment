from __future__ import annotations
import string
from typing import Union, Sequence, Mapping, Any, Tuple
from treelib import Tree, Node
from numbers import Number
from more_itertools import sort_together, split_when, bucket
from itertools import groupby
from collections import namedtuple

import torch


NodeData = namedtuple('NodeData', ['type', 'data'])


def is_primitive(node):
    return (
            isinstance(node, Number)
            or isinstance(node, bool)
            or isinstance(node, str)
            or node is None
    )

def node_type(node):
    if isinstance(node, Number):
        return '___number___'
    elif isinstance(node, bool):
        return '___bool___'
    elif isinstance(node, str):
        return '___string___'
    elif isinstance(node, dict):
        return '___object___'
    elif isinstance(node, list):
        return '___array___'
    elif node is None:
        return '___null___'
    else:
        raise ValueError('node must be of type numbers.Number, bool, str, or NoneType')


def is_array(node):
    return (
            isinstance(node, list)
            or isinstance(node, tuple)
    )


def is_object(node):
    return isinstance(node, dict)


class JSONParseTree(Tree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.all_characters = string.printable
        self.num_characters = len(self.all_characters)

    @classmethod
    def parse_start(cls, root_name: str, node: Any) -> JSONParseTree:
        tree = cls()
        tree.create_node(tag=root_name, identifier=(root_name,), data=NodeData(node_type(node), None))
        if is_array(node):
            for i, child in enumerate(node):
                tree.parse_object((root_name,), str(i), child)
        elif is_object(node):
            for child_name, child in node.items():
                tree.parse_object((root_name,), child_name, child)
        return tree

    def parse_object(self, parent_path: tuple, name: str, node: Any):
        if is_primitive(node):
            self.create_node(
                    tag=name,
                    identifier=parent_path + (name,),
                    parent=parent_path,
                    data=NodeData(node_type(node), node)
            )
        elif is_array(node):
            self.create_node(
                    tag=name,
                    identifier=(new_path := parent_path + (name,)),
                    parent=parent_path,
                    data=NodeData('___array___', None)
            )
            for i, child in enumerate(node):
                self.parse_object(new_path, str(i), child)
        elif is_object(node):
            self.create_node(
                    tag=name,
                    identifier=(new_path := parent_path + (name,)),
                    parent=parent_path,
                    data=NodeData('___dict___', None))
            for child_name, child in node.items():
                self.parse_object(new_path, child_name, child)

    def leaf_data(self) -> Tuple[Tuple, NodeData]:
        for leaf in self.leaves():
            yield leaf.identifier, leaf.data

    def leaf_tensors(self) -> Tuple[Tuple, Any]:
        for leaf in self.leaves():
            if isinstance(leaf.data, Number) or isinstance(leaf.data, bool):
                data = torch.Tensor([[leaf.data]])
            elif isinstance(leaf.data, str):
                data = torch.LongTensor(
                        [self.all_characters.index(char) for char in leaf.data]
                )
            else:
                data = torch.zeros(1, 1)
            yield leaf.identifier, data

    def __eq__(self, other):
        if not isinstance(other, JSONParseTree):
            return False
        self_nodes = sorted(self.nodes)
        other_nodes = sorted(other.nodes)

        return self_nodes == other_nodes

    def __hash__(self):
        return hash(tuple(sorted(self.nodes)))


if __name__ == '__main__':
    from pprint import pprint
    import json

    array = [{"test": {"iest": ['stest', [1, 2, 3]]}, "other": [None, 1, True], "empty": [], "empty_2": {}}] * 3
    some_json = json.loads(r"""
    [{"n": "OO_temp_sensor", "t": 0, "u": "K", "v": 290.02483570765054},
    {"n": "CC_temp_sensor", "t": 0, "u": "K", "v": 290.032384426905},
    {"n": "NW_temp_sensor", "t": 0, "u": "K", "v": 289.98829233126384},
    {"n": "NW_Heater", "t": 0, "u": "W", "v": 185.8732269977827},
    {"n": "NN_temp_sensor", "t": 0, "u": "K", "v": 290.0789606407754},
    {"n": "NN_Heater", "t": 0, "u": "W", "v": 171.3662974759336},
    {"n": "NE_temp_sensor", "t": 0, "u": "K", "v": 289.97652628070324}
    ]
    """)

    trees = [JSONParseTree.parse_start('___root___', arr) for arr in some_json]

    sample_identifiers, sample_index, sample_data = sort_together([*zip(*[
        (leaf_identifier, i, leaf_data) for i, tree in enumerate(trees)
        for leaf_identifier, leaf_data in tree.leaf_tensors()
    ])])

    from prettytable import PrettyTable


    def sample_table(identifiers, index, data):
        table = PrettyTable(('Identifier', 'Batch index', 'Data'))
        for id, idx, dat in zip(identifiers, index, data):
            table.add_row([id, idx, dat])
        print(table)


    sample_table(sample_identifiers, sample_index, sample_data)

    # pprint(list(split_when(zip(sample_identifiers, sample_index, sample_data), lambda x, y: x[0] != y[0])))
    import torch
    buck = {k: [sorted_elem
                if not isinstance(sorted_elem, str) and sorted_elem is not None else sorted_elem
                for sorted_elem in
                zip(*sorted(elem[1:] for elem in g))]
            for k, g in groupby(
                zip(
                        sample_identifiers,
                        sample_index,
                        sample_data
                ), lambda x: x[0]
            )}

    pprint(buck)
