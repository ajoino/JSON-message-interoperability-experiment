from pathlib import Path
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_dir = Path('../tb_logs/without_time_fields/test_results/')
    test_metrics = pd.read_csv(test_dir / 'test_metrics.csv')
    test_actuations = pd.read_csv(test_dir / 'test_actuations.csv')
    test_actuations[['test_pred_actuations', 'test_true_actuations']] *= 333
    test_actuations[['test_pred_actuations', 'test_true_actuations']] += 12

    working_example = test_actuations[
        (test_actuations['mem_dim'] == 64) &
        (test_actuations['size'] == 1) &
        (test_actuations['version'] == 7)
    ]

    print(working_example.head())

    plt.figure()
    sns.kdeplot(x='test_true_actuations', y='test_pred_actuations', hue='size', style='mem_dim', data=test_actuations)
    plt.plot([-300, 300], [-300, 300], 'r')
    plt.show()