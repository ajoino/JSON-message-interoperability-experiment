from typing import Callable, Dict, Any
from collections import defaultdict, namedtuple
import heapq

import torch
from torch import nn
from more_itertools import sort_together

from json_data_module import NUM_CHARACTERS, JSONDataModule, JSON_TYPES, JSON_PRIMITIVES
from json_parser import JSONParseTree


NodeEmbedding = namedtuple('NodeEmbedding', ['memory', 'hidden'])


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


class DefaultModuleDict(nn.ModuleDict):
    def __init__(self, default_factory: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __getitem__(self, item):
        try:
            return super(DefaultModuleDict, self).__getitem__(item)
        except (NameError, KeyError):
            return self.__missing__(item)

    def __missing__(self, key):
        # Taken from json2vec.py
        if self.default_factory is None:
            raise RuntimeError('default_factory is not set')
        else:
            ret = self[key] = self.default_factory()
            return ret

class TypeModule:
    def __init__(self, default_factory: Callable):
        self.default_factory = default_factory

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        if obj not in getattr(obj, self.private_name):
            setattr(obj, self.private_name, nn.ModuleDict())


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, mem_dim: int):
        super().__init__()
        self.mem_dim = mem_dim

        self.childsum_forget = nn.Linear(mem_dim, mem_dim)
        self.childsum_iou = nn.Linear(mem_dim, 3 * mem_dim)

    def forward(self, children_memory: torch.Tensor, children_hidden: torch.Tensor):
        """
        Tensor shape: object_size X sample_indices X embedding_size
        """
        hidden_sum = torch.sum(children_hidden, dim=0)

        forget_gates = torch.sigmoid(self.childsum_forget(children_hidden))
        sigmoid_gates, tanh_gate = torch.split(
                self.childsum_iou(hidden_sum),
                (2 * self.mem_dim, self.mem_dim),
                dim=1
        )
        input_gate, output_gate = torch.split(
                torch.sigmoid(sigmoid_gates),
                (self.mem_dim, self.mem_dim),
                dim=1
        )
        memory_gate = torch.tanh(tanh_gate)
        node_memory = input_gate * memory_gate + torch.sum(forget_gates * children_memory, dim=0)
        node_hidden = output_gate * torch.tanh(node_memory)

        return NodeEmbedding(node_memory, node_hidden)


class JSONLSTMEncoder(nn.Module):
    def __init__(self, mem_dim):
        super().__init__()

        self.mem_dim = mem_dim

        self.__build_model()

    def __build_model(self):
        padding_index = 0
        self.string_embedding = nn.Embedding(
                num_embeddings=NUM_CHARACTERS,
                embedding_dim = self.mem_dim,
                padding_idx=padding_index,
        )
        self.object_lstm = ChildSumTreeLSTM(self.mem_dim)

        self._string_modules = DefaultModuleDict(lambda: nn.LSTM(self.mem_dim, self.mem_dim))
        self._number_modules = DefaultModuleDict(lambda: nn.Linear(1, self.mem_dim))
        self._bool_modules = DefaultModuleDict(lambda: nn.Linear(1, self.mem_dim))
        self._null_modules = DefaultModuleDict(lambda: nn.Linear(1, self.mem_dim))
        self._array_modules = DefaultModuleDict(lambda: nn.LSTM(2 * self.mem_dim, self.mem_dim))
        self._object_modules = DefaultModuleDict(lambda: nn.Linear(self.mem_dim, self.mem_dim))

    def _empty_tensor(self, batch_size):
        return torch.zeros(batch_size, self.mem_dim)

    def forward(self, batch: Dict[str, Any]):
        unique_trees = {tree for keys, data in batch.items() for tree in data['parse_tree']}
        # tree_indices = [i for i, tree in zip(data['sample_index'], data['parse_tree'])]
        if len(unique_trees) == -1:
            """Traverse the tree"""
            tree: JSONParseTree = unique_trees.pop()
            root = tree[tree.root]
            root_type = root.data
            return self.embed_node(root, root_type, batch)

        root = ('___root___',)
        return self.embed_node(root, batch)

    def embed_node(self, node, batch):
        # Find batch node type corresponding to each sample index
        for child_name, child_data in batch.items():
            batch_size = len(child_data['parse_tree'])

            accumulated_memory = self._empty_tensor(batch_size)
            accumulated_hidden = self._empty_tensor(batch_size)
            for type, data in child_data['leaf_data'].items():
                index_tensor = torch.LongTensor([[i for i, tp in enumerate(child_data['type']) if tp == type]]*self.mem_dim).t()
                if type in JSON_PRIMITIVES:
                    result = self.embed_leaf(child_name, type, data)
                accumulated_memory.scatter_(dim=0, index=index_tensor, src=result.memory)
                accumulated_hidden.scatter_(dim=0, index=index_tensor, src=result.hidden)
                a = 1 + 2
            """
            type_indices = {
                node_type: torch.arange(len(data['type']))[
                    torch.BoolTensor([True if type == node_type else False for type in data['type']])]
                for node_type in JSON_TYPES[1:]
            }
            batch_size = len(data['type'])
            accumulated_memory = self._empty_tensor(batch_size)
            accumulated_hidden = self._empty_tensor((batch_size))
            for child_type, child_index in type_indices.items():
                if len(child_index) == 0:
                    continue
                temp_value_tensor = data['leaf_data'].index_select(1, child_index)
                temp_batch = {'type': child_type, 'leaf_data': temp_value_tensor, 'parse_tree': data['parse_tree']}
                if child_type in JSON_PRIMITIVES:
                    child_embeddings = self.embed_leaf(identifier, temp_batch)
                accumulation_mask = torch.IntTensor([[1] if i in child_index  else [0] for i in range(batch_size)])
                accumulated_memory += accumulation_mask * child_embeddings.memory
                accumulated_hidden += accumulation_mask * child_embeddings.hidden

            node_embeddings = NodeEmbedding(accumulated_memory, accumulated_hidden)
            a = 1 + 2
            """

    def embed_leaf(self, identifier, node_type, tensors):
        if node_type == '___string___':
            node_embedding = self.embed_string(tensors, identifier)
        elif node_type == '___number___':
            node_embedding = self.embed_number(tensors, identifier)
        elif node_type == '___bool___':
            node_embedding = self.embed_number(tensors, identifier)
        elif node_type == '___null___':
            node_embedding = self.embed_number(tensors, identifier)
        else:
            raise ValueError(f'node is of unknown type {node_type}')

        return node_embedding

    def embed_object(self, identifier, node_embeddings: NodeEmbedding) -> NodeEmbedding:
        memory, hidden = node_embeddings
        memory, hidden = self.object_lstm(memory, hidden)
        hidden = self._object_modules[str(identifier)](hidden)

        return NodeEmbedding(memory, hidden)

    def embed_array(self, identifier, node_embeddings: NodeEmbedding):
        memory, hidden = node_embeddings


    def embed_string(self, string_batch, key):
        batch_size = string_batch.shape[1]
        string_embeddings = self.string_embedding(string_batch)
        _, (memory, hidden) = self._string_modules[str(key)](string_embeddings)
        return NodeEmbedding(self._empty_tensor(batch_size), hidden.view(batch_size, -1))

    def embed_number(self, number_batch: torch.Tensor, key):
        batch_size = len(number_batch)
        if len(number_batch) > 1:
            # TODO: This is unstable and should be fixed                           vvvvvvvvvvvvvvvvvvvvv
            number_batch = (number_batch - torch.mean(number_batch, dim=0)) / (torch.Tensor((1e-21,)) + torch.std(number_batch, dim=0))

        return NodeEmbedding(self._empty_tensor(batch_size), self._number_modules[str(key)](number_batch))

    def embed_bool(self, bool_batch, key):
        batch_size = len(bool_batch)
        return NodeEmbedding(self._empty_tensor(batch_size), self._bool_modules[str(key)](bool_batch))

    def embed_null(self, null_batch, key):
        batch_size = len(null_batch)
        return NodeEmbedding(self._empty_tensor(batch_size), self._empty_tensor(batch_size))


if __name__ == '__main__':
    data_module = JSONDataModule('../../some_json.json')

    test = JSONLSTMEncoder(128)

    data_module.setup()
    for batch in data_module.train_dataloader():
        print('#### NEW BATCH ####')
        print(test(batch).memory.shape)


    #print([module for module in test.named_modules()])


