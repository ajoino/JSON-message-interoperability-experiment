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
    log_directory = '../tb_logs/with_label_drop_and_multi_pred_layers/'
    run_indices = list(range(4))
    mem_dims = [32]
    path_lengths = [1]
    dropout_rates = [0.5]
    alphas = [0.999]
    label_drop_rates = [0.2, 0.5]
    num_layers = [2]
    batch_size = 50
    train_size = 200000
    val_size = 10000
    max_epochs = 60

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
            [max_epochs],
            [10],
            [train_size],
            [val_size],
            [False],
    ))

    with NestablePool(processes=10) as pool:
        for i in pool.imap(pool_train, parameter_list):
            print(i)
