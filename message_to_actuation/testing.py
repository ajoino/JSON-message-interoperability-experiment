from pathlib import Path
import shutil
import json
from pprint import pprint
import csv
from collections import defaultdict
import itertools
from functools import partial

import torch
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger
from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from datamodule import SimulationDataModule
from network_modules_model_2 import MessageEncoder


def find_top_n_checkpoints(experiments_path: Path, n: int = 3):
    experiment_metrics = pd.read_csv(experiments_path / 'all_metrics.csv')

    groups = experiment_metrics.groupby(['mem_dim', 'prediction_network_size', 'version'])
    #val_metrics = groups[['val_loss', 'val_accuracy', 'val_actuation_loss']]
    val_min = groups['val_loss'].min()
    val_best_versions = val_min.groupby(
            level=[0, 1]
    ).nsmallest(
            n
    ).droplevel(
            level=[0, 1]
    ).reset_index(
    ).drop(
            'val_loss',
            axis='columns'
    )

    return val_best_versions


def get_meta_tags(experiment_dir: Path):
    with open(experiment_dir / 'version_0' / 'meta_tags.csv', 'r') as meta_tags_csv:
        reader = csv.reader(meta_tags_csv)
        meta_tags = dict(row for row in reader)  # type: ignore

    return meta_tags


def get_top_models(experiment_path: Path, top_model_version_df: pd.DataFrame):
    models_dict = defaultdict(lambda: defaultdict(list))
    for experiment_dir in (
            dir for dir in experiment_path.iterdir()
            if (not 'label_loss_drop_rate=0.1' in dir.name) and dir.is_dir() and
            not 'test_results' in dir.name
    ):
        meta_tags = get_meta_tags(experiment_dir)

        experiment_versions = top_model_version_df[
            (top_model_version_df['mem_dim'] == int(meta_tags['mem_dim'])) &
            (top_model_version_df['prediction_network_size'] == int(meta_tags['prediction_network_size']))
        ]['version'].values

        for version in experiment_versions:
            checkpoint_path, *_ = list((experiment_dir / version / 'checkpoints').glob('*.ckpt'))
            model = MessageEncoder.load_from_checkpoint(checkpoint_path)
            #model.eval()
            models_dict[meta_tags['mem_dim']][meta_tags['prediction_network_size']].append((version, model))

    return models_dict


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

    logger = TensorBoardLogger(
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

def test_model_checkpoint(model: MessageEncoder, version: int, experiments_test_path: Path):
    """
    with open(Path(experiment_version_path) / 'meta.experiment', 'r') as meta_file:
        experiment_metadata = json.load(meta_file)
    """

    datamodule = SimulationDataModule(
            Path('../simulation_data_test.csv'),
            50,
            True,
            4,
            test_size=10000,
    )

    logger = TensorBoardLogger(
            str(Path(experiments_test_path)),
            name='test_results',
            version='',
            #description=experiment_metadata['description'] or ''
    )

    model.eval()

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    trainer = Trainer(
            gpus=gpus,
            logger=logger,
            terminate_on_nan=True,
    )

    trainer.test(model, datamodule=datamodule)
    print([name for name in trainer.logged_metrics])
    average_metrics = pd.DataFrame({
            key: [value] for key, value in trainer.logged_metrics.items()
            if key in {'test_accuracy', 'test_mse'}
        } | {'mem_dim': model.mem_dim, 'size': model.prediction_network_size, 'version': version}
    )
    try:
        logged_average_metrics = pd.read_csv(experiments_test_path / 'test_metrics.csv')
        logged_average_metrics = logged_average_metrics.append(average_metrics)
    except FileNotFoundError:
        logged_average_metrics = average_metrics
    logged_average_metrics.to_csv(experiments_test_path / 'test_metrics.csv', index=False)

    timeline_results = pd.DataFrame(
            index=range(0, len(model.test_pred_actuations)),
            data = {
                'timestep': range(0, len(model.test_pred_actuations)),
                'test_pred_actuations': model.test_pred_actuations.flatten(),
                'test_true_actuations': model.test_true_actuations.flatten(),
            }
    )
    timeline_results['mem_dim'] = model.mem_dim
    timeline_results['size'] = model.prediction_network_size
    timeline_results['version'] = version

    print(timeline_results)
    try:
        logged_timeline_results = pd.read_csv(experiments_test_path / 'test_actuations.csv')
        logged_timeline_results = logged_timeline_results.append(timeline_results)
    except FileNotFoundError:
        logged_timeline_results = timeline_results
    logged_timeline_results.to_csv(experiments_test_path / 'test_actuations.csv', index=False)



def model_over_time(model: MessageEncoder, ):
    pass

if __name__ == '__main__':
    experiment_path = Path('../tb_logs/without_time_fields/')
    top_versions = find_top_n_checkpoints(experiment_path)
    top_models = get_top_models(
            experiment_path,
            top_versions,
    )
    for top_models_by_mem_dim in top_models.values():
        for top_models_by_size in top_models_by_mem_dim.values():
            for version, model in top_models_by_size:
                version_num = int(version.split('_')[-1])
                test_model_checkpoint(model, version_num, experiment_path / 'test_results')
