import os
from collections import Sequence
from pathlib import Path
from typing import Any, List, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
import torch.optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from json2vec.json2vec import JSONTreeLSTM

from datamodule import SimulationDataModule

def set_device_in_children(module: nn.Module, device):
    module.device = device

class DynamicParameters:
    def __get__(self, instance, owner):
        return instance.parameters()

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
    ):
        super().__init__()
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
        self.actuation_predict = nn.Sequential(
                nn.Linear(mem_dim + 1, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(30, 1)
        )
        self.label_predict = nn.Sequential(
                nn.Linear(mem_dim, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
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

    def forward(self, messages: Sequence[str], setpoints: torch.Tensor):
        encoded_messages = self.common(torch.cat([self.jsontreelstm(message) for message in messages]))
        actuation_out = self.actuation_predict(torch.cat([encoded_messages, setpoints], dim=1))
        label_out = self.label_predict(encoded_messages)

        return actuation_out, label_out

    def predict_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        #print(room_labels)
        estimated_actuations = self(messages, setpoints)

    def training_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations, estimated_labels = self(messages, setpoints)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        label_loss = self.label_loss(estimated_labels, room_labels.reshape(-1))
        train_loss = actuation_loss + label_loss
        self.log('train_loss', train_loss)
        self.log('actuation_loss', actuation_loss)
        self.log('label_loss', label_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations, estimated_labels = self(messages, setpoints)
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
        estimated_actuations, estimated_labels = self(messages, setpoints)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        label_loss = self.label_loss(estimated_labels, room_labels.reshape(-1))
        test_loss = actuation_loss + label_loss
        return {
            'test_loss': test_loss.reshape(-1),
            'actuation_loss': actuation_loss.reshape(-1),
            'label_loss': label_loss.reshape(-1),
            'pred_labels': torch.argmax(estimated_labels, dim=1),
            'true_labels': room_labels.reshape(-1)
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        testing_metrics = {
            label: torch.cat([metric_dict[label] for metric_dict in outputs]) for label in outputs[0]}
        mean_test_loss = torch.mean(testing_metrics['test_loss'])
        std_test_loss = torch.std(testing_metrics['test_loss'])
        actuation_loss = torch.mean(testing_metrics['actuation_loss'])
        label_loss = torch.mean(testing_metrics['label_loss'])
        test_accuracy = torch.sum(testing_metrics['pred_labels'] == testing_metrics['true_labels']) / len(testing_metrics['pred_labels'])
        self.log('test_loss', mean_test_loss, prog_bar=True)
        self.log('test_loss_std', std_test_loss)
        self.log('test_accuracy', test_accuracy)
        self.log('test_actuation_loss', actuation_loss)
        self.log('test_label_loss', label_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dynamic_parameters)
        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        paths = checkpoint['json_subnetworks']
        for embedder, paths_list in paths.items():
            for paths_str in paths_list:
                getattr(self.jsontreelstm, embedder).add_path(paths_str)
        #print(self.jsontreelstm)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint['json_subnetworks'] = {
            'object_embedder': list(self.jsontreelstm.object_embedder.paths),
            'array_embedder': list(self.jsontreelstm.array_embedder.paths),
            'string_embedder': list(self.jsontreelstm.string_embedder.paths),
            'number_embedder': list(self.jsontreelstm.number_embedder.paths),
        }


if __name__ == '__main__':
    from pprint import pprint

    model = MessageEncoder(mem_dim=128, decode_json=True)
    trainer = Trainer(
            gpus = 1 if torch.cuda.is_available() else None,
            max_epochs=5,
            logger=TestTubeLogger('tb_logs', name='my_model'),
            callbacks=[
                ModelCheckpoint(monitor='val_loss'),
                EarlyStopping(monitor='val_loss', patience=20),
            ],
            resume_from_checkpoint='tb_logs/my_model/version_31/checkpoints/epoch=0-step=4.ckpt',
            fast_dev_run=False,
            terminate_on_nan=True,
            limit_train_batches=5,
            limit_val_batches=1,
            limit_predict_batches=10,
    )

    message_data = SimulationDataModule(Path('../simulation_data_new_all.csv'), batch_size=50, decode_json=True, num_workers=3)
    #trainer.predict(model, datamodule=message_data)
    trainer.fit(model, datamodule=message_data)
    pprint(list(name for name in model.state_dict()))

