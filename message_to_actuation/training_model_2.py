from pathlib import Path
from typing import Tuple, Any, Optional

import pytorch_lightning
import torch
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import SimulationDataModule
from network_modules_model_2 import MessageEncoder

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
        prediction_network_size: int = 1,
        ignore_time_fields: bool = True,
        num_previous_inputs: int = 0,
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
        gpu: Optional[int] = None,
):
    pytorch_lightning.seed_everything((1336 * version + 1), workers=True)
    model = MessageEncoder(
            mem_dim,
            decode_json,
            path_length,
            dropout_rate,
            tie_weights_containers,
            tie_weights_primitives,
            homogeneous_types,
            number_average_fraction,
            prediction_network_size,
            num_previous_inputs,
    )
    datamodule = SimulationDataModule(
            Path('../simulation_data_multi_prev_train.csv'),
            batch_size,
            decode_json,
            data_loader_workers,
            train_size=train_size,
            val_size=val_size,
            ignore_time_fields=ignore_time_fields,
    )

    if gpu is None or not torch.cuda.is_available():
        gpus = None
    else:
        gpus = [gpu]

    experiment_name = (
        f'{experiment_name}-{mem_dim=}-{path_length=}-{dropout_rate=}-'
        f'{number_average_fraction=}-{prediction_network_size=}-{num_previous_inputs=}-'
        f'{batch_size=}-{train_size=}-{val_size=}'
    )

    logger = TestTubeLogger(
            log_directory,
            name=experiment_name,
            version=version if version >= 0 else None,
            description=experiment_description
    )

    trainer = Trainer(
            gpus=None,
            #auto_select_gpus=True,
            max_epochs=max_epochs,
            logger=logger,
            callbacks=[
                ModelCheckpoint(monitor='val_loss'),
                #EarlyStopping(monitor='val_loss', patience=patience),
            ],
            terminate_on_nan=True,
            num_sanity_val_steps=5,
            limit_predict_batches=5,
            progress_bar_refresh_rate=0,
            val_check_interval=0.2,
    )

    trainer.predict(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    return f'Finished training experiment "{experiment_name}" version {version}'

def pool_train(args: Tuple[Any, ...]):
    return train(*args)


if __name__ == '__main__':
    pass
