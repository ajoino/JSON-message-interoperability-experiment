from collections import Sequence
from typing import Any, List, Dict

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn

from json2vec.json2vec import JSONTreeLSTM


def set_device_in_children(module: nn.Module, device):
    module.device = device

class DynamicParameters:
    def __get__(self, instance, owner):
        return instance.parameters()


class ReLUWithDropout(nn.Module):
    def __init__(
            self,
            input_dim: int,
            mem_dim: int,
            output_dim: int,
            num_layers: int,
            dropout_rate: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        if num_layers == 1:
            self.linears = nn.ModuleList(
                    [nn.Linear(input_dim, output_dim)]
            )
        else:
            self.linears = nn.ModuleList(
                    [
                        nn.Linear(input_dim, mem_dim),
                        *(nn.Linear(mem_dim, mem_dim) for _ in range(num_layers - 1)),
                        nn.Linear(mem_dim, output_dim),
                    ]
            )
        self.relus = (nn.ReLU() for _ in range(num_layers))
        self.dropouts = (nn.Dropout() for _ in range(num_layers))

    def forward(self, input: torch.Tensor):
        x = input
        for linear, relu, dropout in zip(self.linears, self.relus, self.dropouts):
            x = dropout(relu(linear(x)))

        return

class MessageEncoder(pl.LightningModule):
    dynamic_parameters = DynamicParameters()

    def __init__(
            self,
            mem_dim: int = 64,
            decode_json: bool = False,
            path_length: int = 1,
            dropout_rate: float = 0.5,
            tie_weights_containers: bool = False,
            tie_weights_primitives: bool = False,
            homogeneous_types: bool = False,
            number_average_fraction: float = 0.999,
            prediction_network_size: int = 1,
            num_previous_actuations: int = 0,
    ):
        super().__init__()
        self.prediction_network_size = int(prediction_network_size)
        self.mem_dim = mem_dim
        self.num_previous_actuations = num_previous_actuations

        self.jsontreelstm = JSONTreeLSTM(
                mem_dim,
                decode_json,
                path_length,
                tie_weights_containers,
                tie_weights_primitives,
                homogeneous_types,
                number_average_fraction,
        )
        self.common = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(2 * mem_dim, mem_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
        )
        # TODO: Add more layers in the prediction subnetworks
        self.actuation_predict = nn.Sequential(
                nn.Linear(mem_dim + 1 + self.num_previous_actuations, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                *[
                    nn.Sequential(
                    nn.Linear(30, 30),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),)
                    for _ in range(self.prediction_network_size - 1)
                ],
                nn.Linear(30, 1)
        )
        self.label_predict = nn.Sequential(
                nn.Linear(mem_dim, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                *[
                    nn.Sequential(
                            nn.Linear(30, 30),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate),)
                    for _ in range(self.prediction_network_size - 1)
                ],
                nn.Linear(30, 8),
        )
        self.label_loss = nn.CrossEntropyLoss()
        self.actuation_loss = nn.MSELoss()

        self.dropout_rate = dropout_rate
        self.train_step = 0

        self.save_hyperparameters(ignore=['train_step'])

    def on_train_start(self):
        self.train_step = 0
        self.jsontreelstm.apply(lambda m: set_device_in_children(m, self.device))

    def on_train_batch_end(self, *args, **kwargs):
        self.train_step += 1

    def on_validation_start(self):
        self.jsontreelstm.apply(lambda m: set_device_in_children(m, self.device))

    def on_validation_epoch_end(self):
        self.train_step += 1

    def on_predict_start(self):
        self.jsontreelstm.apply(lambda m: set_device_in_children(m, self.device))

    def on_test_start(self):
        self.jsontreelstm.apply(lambda m: set_device_in_children(m, self.device))
        #self.jsontreelstm.register_forward_hook(pass)

    def forward(self, messages: Sequence[str], setpoints: torch.Tensor, previous_actuations: torch.Tensor):
        encoded_messages = self.common(torch.cat([
            self.jsontreelstm(message) for message in messages
        ]))
        actuation_out = self.actuation_predict(torch.cat([
            encoded_messages,
            setpoints,
            previous_actuations[:, 0:self.num_previous_actuations]
        ], dim=1))
        label_out = self.label_predict(encoded_messages)

        return actuation_out, label_out

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        messages, room_labels, setpoints, actuations, prev_actuations = batch

        return self(messages, setpoints, prev_actuations)

    def training_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations, estimated_labels = self(messages, setpoints, prev_actuations)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        label_loss = self.label_loss(estimated_labels, room_labels.reshape(-1))
        train_loss = actuation_loss + label_loss
        self.log('train_loss', train_loss)
        self.log('actuation_loss', actuation_loss)
        self.log('label_loss', label_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations, estimated_labels = self(messages, setpoints, prev_actuations)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        label_loss = self.label_loss(estimated_labels, room_labels.reshape(-1))
        val_loss = actuation_loss + label_loss
        return {
            'val_loss': val_loss.reshape(-1),
            'actuation_loss': actuation_loss.reshape(-1),
            'label_loss': label_loss.reshape(-1),
            'pred_labels': torch.argmax(estimated_labels, dim=1),
            'true_labels': room_labels.reshape(-1)
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        validation_metrics = {
            label: torch.cat([metric_dict[label] for metric_dict in outputs]) for label in outputs[0]}
        mean_val_loss = torch.mean(validation_metrics['val_loss'])
        std_val_loss = torch.std(validation_metrics['val_loss'])
        actuation_loss = torch.mean(validation_metrics['actuation_loss'])
        label_loss = torch.mean(validation_metrics['label_loss'])
        val_accuracy = torch.sum(validation_metrics['pred_labels'] == validation_metrics['true_labels']) / len(validation_metrics['pred_labels'])
        self.log('val_loss', mean_val_loss, prog_bar=True)
        self.log('val_loss_std', std_val_loss)
        self.log('val_accuracy', val_accuracy)
        self.log('val_actuation_loss', actuation_loss)
        self.log('val_label_loss', label_loss)

    def test_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations, estimated_labels = self(messages, setpoints, prev_actuations)
        return {
            'pred_actuations': estimated_actuations,
            'true_actuations': actuations,
            'pred_labels': torch.argmax(estimated_labels, dim=1),
            'true_labels': room_labels.reshape(-1)
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        testing_metrics: Dict[str, torch.Tensor] = {
            label: torch.cat([metric_dict[label] for metric_dict in outputs]) for label in outputs[0]
        }
        test_mse = torch.mean((testing_metrics['pred_actuations'] - testing_metrics['true_actuations'])**2)
        test_accuracy = torch.sum(testing_metrics['pred_labels'] == testing_metrics['true_labels']) / len(testing_metrics['pred_labels'])
        self.log('test_accuracy', test_accuracy)
        self.log('test_mse', test_mse)
        self.test_pred_actuations = testing_metrics['pred_actuations'].detach().cpu().numpy()
        self.test_true_actuations = testing_metrics['true_actuations'].detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dynamic_parameters, weight_decay=0.0001)
        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        paths = checkpoint['json_subnetworks']
        for embedder, paths_list in paths.items():
            for paths_str in paths_list:
                getattr(self.jsontreelstm, embedder).add_path(paths_str)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint['json_subnetworks'] = {
            'object_embedder': list(self.jsontreelstm.object_embedder.paths),
            'array_embedder': list(self.jsontreelstm.array_embedder.paths),
            'string_embedder': list(self.jsontreelstm.string_embedder.paths),
            'number_embedder': list(self.jsontreelstm.number_embedder.paths),
        }

