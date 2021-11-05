from pathlib import Path
from pprint import pprint

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_epoch(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics['progress'] = ''
    metrics['progress'][~metrics['train_loss'].isna()] = 'train'
    metrics['progress'][~metrics['val_loss'].isna()] = 'val'
    train_step = metrics.groupby(by=['epoch', 'progress']).cumcount() + 1
    epoch_max = train_step.max()
    metrics['progress'] = 0.0
    metrics['progress'][~metrics['train_loss'].isna()] = train_step / epoch_max
    metrics['progress'][~metrics['val_loss'].isna()] = train_step / 5
    metrics['progress'] += metrics['epoch']

    return metrics


def generate_tables(experiment_parent_path: Path) -> pd.DataFrame:
    experiment_metrics = pd.DataFrame()
    for experiment_path in experiment_parent_path.expanduser().iterdir():
        if not experiment_path.is_dir():
            continue
        experiment_meta_tags = {
            meta_tag: meta_value
            for meta_tag, meta_value
            in pd.read_csv(experiment_path / 'version_0/meta_tags.csv').values
            if meta_tag in {
                'dropout_rate',
                'label_loss_drop_rate',
                'mem_dim',
                'number_average_fraction',
                'path_length',
                'prediction_network_size'
            }
        }
        for version_path in experiment_path.iterdir():
            metrics = pd.read_csv(version_path / 'metrics.csv')
            metrics['version'] = version_path.name
            metrics = calculate_epoch(metrics)
            for meta_tag, meta_value in experiment_meta_tags.items():
                metrics[meta_tag] = meta_value
            experiment_metrics = experiment_metrics.append(metrics)

    return experiment_metrics


def plot_training_data(metrics: pd.DataFrame, max_epoch: int = 40):
    plt.figure()
    print('Creating training figure loss')
    sns.lineplot(y='train_loss', x='progress', hue='mem_dim', style='prediction_network_size', data=metrics[metrics['epoch'] <= max_epoch], ci=None)
    #sns.lineplot(y='actuation_loss', x='progress', hue='mem_dim', data=metrics, label='Training actuation loss', ci=95, n_boot=10)
    #sns.lineplot(y='label_loss', x='progress', hue='mem_dim', data=metrics, label='Training label loss', ci=95, n_boot=10)
    plt.ylabel('Loss')
    plt.ylim((0, 2.5))
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/train_loss.pdf')


def plot_validation_data(metrics: pd.DataFrame, max_epoch: int = 40):
    plt.figure()
    sns.lineplot(y='val_loss', x='progress', hue='mem_dim', style='prediction_network_size', data=metrics[metrics['epoch'] <= max_epoch], ci=None)
    plt.ylabel('Validation Loss')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/val_loss.pdf')

    plt.figure()
    sns.lineplot(y='val_accuracy', x='progress', hue='mem_dim', style='prediction_network_size', data=metrics[metrics['epoch'] <= max_epoch], ci=None)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/val_acc.pdf')

    plt.figure()
    sns.lineplot(y='val_actuation_loss', x='progress', hue='mem_dim', style='prediction_network_size', data=metrics[metrics['epoch'] <= max_epoch], ci=None)
    plt.ylabel('Validation Actuation MSE')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/val_actuation_loss.pdf')

if __name__ == '__main__':
    experiment_path = Path('~/articles/spring_2021_experiment/tb_logs/without_time_fields/')
    experiment_metrics = generate_tables(experiment_path)
    print(experiment_metrics.shape)
    print(experiment_metrics.columns)
    #experiment_metrics = experiment_metrics[experiment_metrics['label_loss_drop_rate'] == '0.0']
    print(experiment_metrics.shape)

    experiment_metrics.to_csv(experiment_path / 'all_metrics.csv')

    plot_training_data(experiment_metrics)
    plot_validation_data(experiment_metrics, 60)
    plt.show()

