import os
from collections import Sequence
from pathlib import Path
from typing import Any, List

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn

from json2vec.json2vec import JSONTreeLSTM

from datamodule import SimulationDataModule

class MessageEncoder(pl.LightningModule):
    def __init__(self, mem_dim: int = 64):
        super().__init__()
        self.jsontreelstm = JSONTreeLSTM(mem_dim)
        self.compress_tree = nn.Linear(2 * mem_dim, 10)
        self.output = nn.Linear(11, 1)

    def on_train_start(self):
        self.jsontreelstm.device = self.device

    def on_validation_start(self):
        self.jsontreelstm.device = self.device

    def forward(self, messages: Sequence[str], setpoints: torch.Tensor):
        encoded_messages = torch.relu(torch.cat([self.jsontreelstm(message) for message in messages]))
        compressed_encodings = torch.cat([torch.relu(self.compress_tree(encoded_messages)), setpoints], dim=1)

        return self.output(compressed_encodings)

    def training_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations = batch
        estimated_actuations = self(messages, setpoints)
        train_loss = torch.nn.functional.mse_loss(estimated_actuations, actuations)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        messages, room_labels, setpoints, actuations = batch
        estimated_actuations = self(messages, setpoints)
        val_loss = torch.nn.functional.mse_loss(estimated_actuations, actuations)
        return val_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mean_val_loss = torch.mean(torch.column_stack(outputs))
        self.log('val_loss', mean_val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    model = MessageEncoder()
    trainer = Trainer(
	    gpus=1,
            max_epochs=4,
            callbacks=[
                ModelCheckpoint(monitor='val_loss'),
                EarlyStopping(monitor='val_loss'),
            ],
    )

    message_data = SimulationDataModule(Path('../simulation_data.csv'), batch_size=32)
    trainer.fit(model, message_data)

