import os
from collections import Sequence
from pathlib import Path
from typing import Any, List

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from json2vec.json2vec import JSONTreeLSTM

from datamodule import SimulationDataModule

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
        self.dropout_1 = nn.Dropout()
        self.compress_tree = nn.Linear(2 * mem_dim, mem_dim)
        self.dropout_2 = nn.Dropout()
        self.actuation_intermediate = nn.Linear(mem_dim + 1, 30)
        self.dropout_3 = nn.Dropout()
        self.actuation_output = nn.Linear(30, 1)

        self.label_loss = nn.CrossEntropyLoss()
        self.actuation_loss = nn.MSELoss()

    def on_train_start(self):
        self.jsontreelstm.device = self.device

    def on_validation_start(self):
        self.jsontreelstm.device = self.device

    def forward(self, messages: Sequence[str], setpoints: torch.Tensor):
        encoded_messages = self.dropout_1(torch.relu(torch.cat([self.jsontreelstm(message) for message in messages])))
        #encodings_cat = torch.cat([encoded_messages, setpoints], dim=1)
        modality_mix = self.dropout_2(torch.relu(self.compress_tree(encoded_messages)))
        actuation_inter = self.dropout_3(torch.relu(self.actuation_intermediate(torch.cat([modality_mix, setpoints], dim=1))))
        #room_name_inter = torch.relu(self.room_name_intermediate(modality_mix))

        return self.actuation_output(actuation_inter)#, self.room_name_output(room_name_inter)

    def predict_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations = self(messages, setpoints)

    def training_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations = self(messages, setpoints)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        #train_acc = torch.sum(torch.argmax(estimated_labels, dim=1) == room_labels.flatten()) / len(room_labels)
        #jlabel_loss = self.label_loss(estimated_labels, room_labels.flatten())
        train_loss = actuation_loss# + label_loss
        self.log('train_loss', train_loss)
        self.log('mse_loss', actuation_loss, prog_bar=True)
        #self.log('acc', train_acc, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations, prev_actuations = batch
        estimated_actuations = self(messages, setpoints)
        actuation_loss = self.actuation_loss(estimated_actuations, actuations)
        #val_acc = torch.sum(torch.argmax(estimated_labels, dim=1) == room_labels.flatten()) / len(room_labels)
        #label_loss = self.label_loss(estimated_labels, room_labels.flatten())
        val_loss = actuation_loss# + label_loss
        return val_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_loss = outputs
        #mean_val_acc = torch.mean(torch.column_stack(val_acc))
        mean_val_loss = torch.mean(torch.column_stack(val_loss))
        self.log('val_loss', mean_val_loss, prog_bar=True)
        #self.log('val_acc', mean_val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dynamic_parameters)
        return {
            'optimizer': optimizer,
            #'lr_scheduler': {
            #    'scheduler': CyclicLR(optimizer, 0.001, 0.2),
            #    'monitor': 'val_loss',
            #}
        }


if __name__ == '__main__':
    model = MessageEncoder(mem_dim=128, decode_json=True)
    trainer = Trainer(
            gpus = 1 if torch.cuda.is_available() else None,
            max_epochs=200,
            callbacks=[
                ModelCheckpoint(monitor='train_loss'),
                EarlyStopping(monitor='train_loss', patience=20),
            ],
            fast_dev_run=False,
            terminate_on_nan=True,
            overfit_batches=10,
            limit_predict_batches=10,
            #val_check_interval=10,
            check_val_every_n_epoch=10,
            num_sanity_val_steps=2,
    )

    message_data = SimulationDataModule(Path('../simulation_data_new_all.csv'), batch_size=50, decode_json=True, num_workers=3)
    trainer.predict(model, datamodule=message_data)
    trainer.fit(model, datamodule=message_data)
    print(list(name for name in model.state_dict()))

