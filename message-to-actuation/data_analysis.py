from pathlib import Path
import getpass

import pysftp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_logs(host: str, user: str, remote_dir: str, target_dir: str):
    password = getpass.getpass(f'{user}@{host}\'s password: ')
    if not Path(target_dir).exists():
        Path(target_dir).mkdir()

    with pysftp.Connection(host, username=user, password=password) as sftp:
        sftp.chdir(remote_dir)
        sftp.get_r('.', target_dir)


def handle_metrics(version_dir: Path):
    df = pd.read_csv(version_dir / 'metrics.csv')
    df['version'] = version_dir.name
    df['process'] = ''
    df['process'][~df['train_loss'].isna()] = 'train'
    df['process'][~df['val_loss'].isna()] = 'val'
    train_step = df.groupby(by=['epoch', 'process']).cumcount() + 1
    epoch_max = train_step.max()
    df['progress'] = 0.0
    df['progress'][~df['train_loss'].isna()] = train_step / epoch_max
    df['progress'][~df['val_loss'].isna()] = train_step / 5
    df['progress'] += df['epoch']

    return df


if __name__ == '__main__':
    df_list = [handle_metrics(version_dir) for version_dir in (
                Path.cwd().parent.parent / 'remote_tb_logs/jsontreelstm-pl=1-md=32-dr=0.5-alpha=0.999-batch_size=50-train_size=200000-val_size=10000').iterdir()]
    df = pd.concat(df_list, ignore_index=True)
    del df_list

    figure_dir = Path.cwd().parents[1] / 'figures'
    if not figure_dir.is_dir():
        figure_dir.mkdir()

    plt.figure()
    print('Creating training figure loss')
    sns.lineplot(y='train_loss', x='progress', data=df, label='Training loss')
    sns.lineplot(y='actuation_loss', x='progress', data=df, label='Training actuation loss')
    sns.lineplot(y='label_loss', x='progress', data=df, label='Training label loss')
    plt.ylabel('Loss')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/train_loss.pdf')

    plt.figure()
    print('Creating validation figure loss')
    sns.lineplot(y='val_loss', x='progress', data=df, label='Validation loss')
    sns.lineplot(y='val_actuation_loss', x='progress', data=df, label='Validation actuation loss')
    sns.lineplot(y='val_label_loss', x='progress', data=df, label='Validation label loss')
    plt.ylabel('Loss')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/val_loss.pdf')

    plt.figure()
    print('Creating validation figure accuracy')
    sns.lineplot(y='val_accuracy', x='progress', data=df)
    plt.ylabel('Room prediction accuracy')
    plt.xlabel('Progress (epochs)')
    plt.savefig('../../figures/val_acc.pdf')
