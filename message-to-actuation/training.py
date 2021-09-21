from pathlib import Path
from typing import Sequence, Tuple, Any
import itertools
import datetime

import torch
from torch.multiprocessing import Pool
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule import SimulationDataModule
from network_modules import MessageEncoder

def train(
        # model parameters
        mem_dim: int = 128,
        decode_json: bool = True,
        path_length: int = 1,
        dropout_rate: float = 0.5,
        tie_weights_containers: bool = False,
        tie_weights_primitives: bool = False,
        homogeneous_types: bool = False,
        number_average_fraction: float = 0.999,
        label_loss_drop_rate: float = 0.75,
        prediction_network_size: int = 1,
        # experiment parameters
        batch_size: int = 50,
        data_loader_workers: int = 4,
        log_directory: str = Path.cwd() / 'tb_logs',
        experiment_name: str = 'model',
        experiment_description: str = '',
        version: int = -1,
        max_epochs: int = 200,
        patience: int = 20,
        train_size: int = 55000,
        val_size: int = 5000,
        force_cpu: bool = False,
):
    model = MessageEncoder(
            mem_dim,
            decode_json,
            path_length,
            dropout_rate,
            tie_weights_containers,
            tie_weights_primitives,
            homogeneous_types,
            number_average_fraction,
            label_loss_drop_rate,
            prediction_network_size
    )
    datamodule = SimulationDataModule(
            Path('../simulation_data_train.csv'),
            batch_size,
            decode_json,
            data_loader_workers,
            train_size=train_size,
            val_size=val_size,
    )

    if force_cpu or not torch.cuda.is_available():
        gpus = None
    else:
        gpus = 1

    experiment_name = (
        f'{experiment_name}-{mem_dim=}-{path_length=}-{dropout_rate=}-'
        f'{number_average_fraction=}-{label_loss_drop_rate=}-'
        f'{prediction_network_size=}-{batch_size=}-{train_size=}-'
        f'{val_size=}'
    )

    logger = TestTubeLogger(
            log_directory,
            name=experiment_name,
            version=version if version >= 0 else None,
            description=experiment_description
    )

    trainer = Trainer(
            gpus=[2],
            #auto_select_gpus=True,
            max_epochs=max_epochs,
            logger=logger,
            callbacks=[
                ModelCheckpoint(monitor='val_loss'),
                EarlyStopping(monitor='val_loss', patience=patience),
            ],
            terminate_on_nan=True,
            num_sanity_val_steps=5,
            limit_predict_batches=5,
            progress_bar_refresh_rate=0,
            val_check_interval=0.2,
    )

    trainer.predict(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    return f'Finished training experiment "{experiment_name}" version "version"'

def pool_train(args: Tuple[Any, ...]):
    return train(*args)


if __name__ == '__main__':
    pass
