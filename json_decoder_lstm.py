from enum import Enum

import torch
from torch import nn


class JSONConstructionTokens(int, Enum):
    STOP = 0
    START = 1
    OBJ_BEGIN = 2
    OBJ_END = 3
    ARR_BEGIN = 4
    ARR_END = 5
    STRING = 6
    NUMBER = 7
    BOOL = 8
    NULL = 9


def valid_token_mask(input_node):
    token_dictionary = {
        JSONConstructionTokens.START: torch.Tensor([0.0 if i not in {1, 3, 5} else -float('inf') for i in range(10)]),
        JSONConstructionTokens.STOP: torch.Tensor([0.0 if i == 0 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.OBJ_BEGIN: torch.Tensor(
                [0.0 if i not in {0, 1, 5} else -float('inf') for i in range(10)]),
        JSONConstructionTokens.OBJ_END: torch.Tensor([0.0 if i != 0 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.ARR_BEGIN: torch.Tensor(
                [0.0 if i not in {0, 1, 3} else -float('inf') for i in range(10)]),
        JSONConstructionTokens.ARR_END: torch.Tensor([0.0 if i != 1 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.STRING: torch.Tensor([0.0 if i != 1 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.NUMBER: torch.Tensor([0.0 if i != 1 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.BOOL: torch.Tensor([0.0 if i != 1 else -float('inf') for i in range(10)]),
        JSONConstructionTokens.NULL: torch.Tensor([0.0 if i != 1 else -float('inf') for i in range(10)]),
    }
    return torch.cat([token_dictionary[int(input)] for input in input_node.t()]).view(-1, 10)


def bracket_mask(bracket_stack, current_node):
    for stack, node_value in zip(bracket_stack, current_node.t()):
        if ((nd := int(node_value)) == JSONConstructionTokens.OBJ_BEGIN
                or nd == JSONConstructionTokens.ARR_BEGIN):
            stack.append(nd)
        elif (stack[-1] == JSONConstructionTokens.OBJ_BEGIN and int(node_value) == JSONConstructionTokens.OBJ_END):
            stack.pop()
        elif (stack[-1] == JSONConstructionTokens.ARR_BEGIN and int(node_value) == JSONConstructionTokens.ARR_END):
            stack.pop()
    mask = valid_token_mask(torch.Tensor([stack[-1] for stack in bracket_stack]))
    return mask




def change_node_level(current_node):
    change_list = [0, 0, 1, -1, 1, -1, 0, 0, 0, 0]
    return torch.LongTensor([change_list[int(node)] for node in current_node])


def has_stopped_mask(current_node_level):
    return torch.LongTensor([1 if level > 0 else 0 for level in current_node_level[0]])


class JSONStructureDecoder(nn.Module):
    def __init__(self, mem_dim: int, max_seq_len: int = 256):
        super().__init__()

        self.mem_dim = mem_dim
        self.max_seq_len = max_seq_len
        self.input_size = len(JSONConstructionTokens)

        self.node_embedder = nn.Embedding(self.input_size, self.mem_dim)
        self.generator_lstm = nn.LSTM(self.mem_dim, self.mem_dim)
        self.node_predictor = nn.Linear(self.mem_dim, self.input_size)

    def forward(self, memory: torch.Tensor, hidden: torch.Tensor):
        batch_size = memory.shape[1]
        has_stopped = torch.ones(1, batch_size, dtype=torch.long)
        current_node_level = torch.zeros(1, batch_size, dtype=torch.long)
        bracket_stack = [[0] for _ in range(batch_size)]
        b_mask = torch.zeros(batch_size, self.input_size)
        combined_generator_outputs = []
        for i in range(self.max_seq_len):
            if torch.all(has_stopped == 0):
                break
            if i == 0:
                current_node = torch.ones(1, batch_size, dtype=torch.long)
            else:
                current_node = torch.cat((current_node, predicted_node), dim=0)
                b_mask = bracket_mask(bracket_stack, current_node[-1, :])

            embedded_input = self.node_embedder(current_node[-1:, :])
            generator_output, (hidden, memory) = self.generator_lstm(embedded_input, (hidden, memory))
            raw_output = self.node_predictor(generator_output)
            current_node_level += change_node_level(current_node[-1, :])
            masked_output = (raw_output
                             + valid_token_mask(current_node[-1, :])
                             + b_mask)
            predicted_node = torch.argmax(masked_output, dim=-1).view(1, -1)

            combined_generator_outputs.append(generator_output)

            if i > 0:
                predicted_node = predicted_node * has_stopped_mask(current_node_level)

            if i > 1:
                has_stopped = predicted_node

            return (current_node, masked_output)
