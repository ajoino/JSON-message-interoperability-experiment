import itertools
from typing import Sequence
import multiprocessing
import multiprocessing.pool


import torch
#from torch.multiprocessing import Pool, Process, set_start_method

from training import train, pool_train

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

def start_training_processes(
        experiment_name: str,
        log_directory: str,
        num_runs: int = 1,
        mem_dims: Sequence[int] = None,
        path_lengths: Sequence[int] = None,
        dropout_rates: Sequence[float] = None,
        alphas: Sequence[float] = None,
):
    run_indices = list(range(num_runs))
    mem_dims = mem_dims or [32]
    path_lengths = path_lengths or [1]
    dropout_rates = dropout_rates or [0.5]
    alphas = alphas or [0.999]
    batch_size = 50
    train_size = 200000
    val_size = 10000

    parameter_list = itertools.product(
            mem_dims,
            [True],
            path_lengths,
            dropout_rates,
            [False], [False], [False],
            alphas,
            [batch_size],
            [10],
            [log_directory],
            [experiment_name],
            [''],
            run_indices,
            [200],
            [20],
            [train_size],
            [val_size],
            [False],
    )
    # model parameters
    #mem_dim: int = 128,
    #decode_json: bool = True,
    #path_length: int = 1,
    #dropout_rate: float = 0.5,
    #tie_weights_containers: bool = False,
    #tie_weights_primitives: bool = False,
    #homogeneous_types: bool = False,
    #number_average_fraction: float = 0.999,
    ## experiment parameters
    #batch_size: int = 50,
    #data_loader_workers: int = 4,
    #log_directory: str = Path.figure_dir() / 'tb_logs',
    #experiment_name: str = 'model',
    #experiment_description: str = '',
    #version: int = -1,
    #max_epochs: int = 200,
    #patience: int = 20,
    #train_size: int = 55000,
    #val_size: int = 5000,
    #force_cpu: bool = False,

    with Pool(processes=4) as pool:
        pool.starmap(train, parameter_list)

if __name__ == '__main__':
    log_directory = '../tb_logs/with_label_drop_and_multi_pred_layers/'
    run_indices = list(range(4))
    mem_dims = [32]
    path_lengths = [1]
    dropout_rates = [0.5]
    alphas = [0.999]
    label_drop_rates = [0.2, 0.8]
    num_layers = [10]
    batch_size = 50
    train_size = 200
    val_size = 50

    experiment_names = [
        'jsontreelstm-' + f'{pl=}-' + f'{md=}-' + f'{label_dr=}-' + f'{nl=}-' + f'{dr=}-' + f'{alpha=}-' + f'{batch_size=}-' + f'{train_size=}-' + f'{val_size=}'
        for pl, md, label_dr, nl, dr, alpha in zip(path_lengths, mem_dims, label_drop_rates, num_layers, dropout_rates, alphas)
    ]

    parameter_list = list(itertools.product(
            mem_dims,
            [True],
            path_lengths,
            dropout_rates,
            [False], [False], [False],
            alphas,
            label_drop_rates,
            num_layers,
            [batch_size],
            [10],
            [log_directory],
            ['jsontreelstm'],
            [''],
            run_indices,
            [1],
            [20],
            [train_size],
            [val_size],
            [False],
    ))
    from pprint import pprint

    with NestablePool(processes=5) as pool:
        for i in pool.imap(pool_train, parameter_list):
            print(i)
