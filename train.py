import argparse

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np

from chainercv.datasets.cub.cub_label_dataset import CUBLabelDataset
from chainer.datasets import TransformDataset

from chainer.links.model.vision.vgg import prepare


class TripletDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.labels = self.dataset.slice[:, 'label']

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        x_a, l_a = self.dataset[i]
        x_p_idx = np.random.choice(np.where(self.labels == l_a)[0])
        x_n_idx = np.random.choice(np.where(self.labels != l_a)[0])
        x_p = self.dataset[x_p_idx][0]
        x_n = self.dataset[x_n_idx][0]
        return x_a, x_p, x_n


class FinetuneVGG(chainer.Chain):

    def __init__(self):
        super(FinetuneVGG, self).__init__()
        with self.init_scope():
            self.encoder = L.VGG16Layers(pretrained_model='auto')

    def forward(self, x):
        chainer.report({'loss': 1}, self)
        return self.encoder(x, layers=['fc6'])['fc6']

    def __call__(self, x_a, x_p, x_n):
        z_a, z_p, z_n = self.forward(x_a), self.forward(x_p), self.forward(x_n)
        loss = F.triplet(z_a, z_p, z_n)
        chainer.report({'loss': loss}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Chainer example: Triplet CUB')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = args.gpu

    # Set up a neural network to train

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    model = FinetuneVGG()
    optimizer.setup(model)
    model.to_gpu(device)

    def get_idx(train=True):
        path = 'CUB_200_2011/train_test_split.txt'
        with open(path, 'rt') as rf:
            mask = [int(line.split(' ')[1]) for line in rf.read().strip().split('\n')]
            if train:
                idx = [x == 0 for x in mask]
            else:
                idx = [x == 1 for x in mask]
        return idx

    def transform(in_data):
        x_a, x_p, x_n = in_data
        return prepare(x_a), prepare(x_p), prepare(x_n)

    train_core = CUBLabelDataset()
    train_idx = get_idx()
    train = TransformDataset(TripletDataset(train_core, train_idx), transform)

    # test_core = CUBLabelDataset()
    # test_idx = get_idx(False)
    # test = TripletDataset(test_core, test_idx)

    train_iter = chainer.iterators.SerialIterator(train, 32)
    # test_iter = chainer.iterators.SerialIterator(test, 32, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (20000, 'iteration'))

    # きもすぎ
    trainer.extend(
        extensions.LogReport([
            'iteration',
            'main/loss',
            'elapsed_time'
        ], trigger=(1, 'iteration'))
    )
    trainer.extend(
        extensions.PrintReport([
            'iteration',
            'main/loss',
            'elapsed_time'
        ]),
        trigger=(1, 'iteration')
    )
    trainer.extend(training.extensions.ProgressBar(update_interval=1))
    trainer.run()


if __name__ == '__main__':
    main()
