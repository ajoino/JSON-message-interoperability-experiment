import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import sklearn
pl.LightningDataModule()

from json2vec import JSONTreeLSTM
from datasets import load_seismic_dataset


class SeismicDataset(Dataset):
    """Seismic dataset"""

    def __init__(self):
        jsons, vectors, labels = load_seismic_dataset()
        labels = torch.LongTensor([int(label) for label in labels])
        self.jsons, self.vectors, self.labels = sklearn.utils.shuffle(jsons, vectors, labels)

    def __len__(self):
        return len(self.jsons)

    def __getitem__(self, idx):
        return self.jsons[idx], self.labels[idx]


class JsonTreeSystem(pl.LightningModule):

    def __init__(self, mem_dim=128):
        super().__init__()

        self.json_tree_lstm = JSONTreeLSTM(mem_dim=mem_dim)
        self.output = nn.Linear(2 * mem_dim, 1)

    def forward(self, *args, **kwargs):
        return self.json_tree_lstm(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        jsons, labels = batch
        labels = torch.LongTensor([int(label) for label in labels])
        output = self.json_tree_lstm(*jsons)
        output = torch.sigmoid(self.output(output).view(1))

        loss = F.binary_cross_entropy(output, labels.float())

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    from pprint import pprint
    seismic_dataset = SeismicDataset()
    train_loader = DataLoader(seismic_dataset)
    json_tree = JsonTreeSystem(128)
    trainer = pl.Trainer(overfit_batches=1)
    trainer.fit(json_tree, train_loader)