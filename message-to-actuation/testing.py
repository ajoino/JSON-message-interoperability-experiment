from pathlib import Path
import json

import torch
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule import SimulationDataModule
from network_modules import MessageEncoder

def str_to_parameter_type(parameter_str: str):
    if parameter_str == 'True':
        return True
    elif parameter_str == 'False':
        return False
    elif parameter_str == 'None':
        return None
    else:
        try:
            return int(parameter_str)
        except ValueError:
            try:
                return float(parameter_str)
            except ValueError:
                return parameter_str


def test_single_checkpoint(
        experiment_version_path: str,
        batch_size: int = 50,
        test_size: int = 100,
        data_loader_workers: int = 4,
        decode_json: bool = True,
        force_cpu: bool = False,
):
    with open(Path(experiment_version_path) / 'meta.experiment', 'r') as meta_file:
        experiment_metadata = json.load(meta_file)

    datamodule = SimulationDataModule(
            Path('../simulation_data_test.csv'),
            batch_size,
            decode_json,
            data_loader_workers,
            test_size=test_size,
    )

    if force_cpu or not torch.cuda.is_available():
        gpus = None
    else:
        gpus = 1

    logger = TestTubeLogger(
            str(Path(experiment_version_path).parent.parent),
            name=experiment_metadata['name'],
            version=experiment_metadata['version'],
            description=experiment_metadata['description'] or ''
    )

    checkpoint_path, *_ = list((Path(experiment_version_path) / 'checkpoints').glob('*.ckpt'))
    model = MessageEncoder.load_from_checkpoint(checkpoint_path)
    model.eval()

    trainer = Trainer(
            gpus=gpus,
            logger=logger,
            terminate_on_nan=True,
    )

    trainer.test(model, datamodule=datamodule)

def test_all_checkpoints(
        model_path: str,
        batch_size: int = 50,
        test_size: int = 1000,
        data_loader_workers: int = 4,
        decode_json: bool = True,
        force_cpu: bool = False,
):
    versions_paths = [str(f) for f in Path(model_path).iterdir() if f.is_dir() and f.name.startswith('version')]

    for version_path in sorted(versions_paths, key=lambda x: int(Path(x).name.split('_')[-1])):
        try:
            test_single_checkpoint(
                    version_path,
                    batch_size,
                    test_size,
                    data_loader_workers,
                    decode_json,
                    force_cpu,
            )
        except (RuntimeError, ValueError):
            continue

if __name__ == '__main__':
    test_all_checkpoints('tb_logs/model')