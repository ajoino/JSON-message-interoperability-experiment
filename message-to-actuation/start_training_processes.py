import itertools
import multiprocessing.pool

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


if __name__ == '__main__':
    log_directory = '../tb_logs/without_time_fields/'
    run_indices = list(range(1))
    mem_dims = [32]
    path_lengths = [1]
    dropout_rates = [0.5]
    alphas = [0.999]
    label_drop_rates = [0.0]
    num_layers = [2]
    ignore_time_fields = True
    batch_size = 50
    train_size = 200
    val_size = 50
    max_epochs = 1

    parameter_list = list(itertools.product(
            mem_dims,
            [True],
            path_lengths,
            dropout_rates,
            [False], [False], [False],
            alphas,
            num_layers,
            [ignore_time_fields],
            [batch_size],
            [10],
            [log_directory],
            ['jsontreelstm'],
            [''],
            run_indices,
            [max_epochs],
            [10],
            [train_size],
            [val_size],
            [False],
    ))

    with NestablePool(processes=10) as pool:
        for i in pool.imap(pool_train, parameter_list):
            print(i)
