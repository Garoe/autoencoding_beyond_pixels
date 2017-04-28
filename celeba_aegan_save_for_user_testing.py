#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy as sp
import deeppy as dp
import output

import dataset.celeba
import aegan
from dataset.util import img_transform, img_inverse_transform


def run():
    experiment_name = 'celeba'

    img_size = 64
    epoch_size = 250
    batch_size = 2000

    n_hidden = 128
    _, experiment_name = aegan.build_model(
        experiment_name, img_size, n_hidden=n_hidden, recon_depth=9,
        recon_vs_gan_weight=1e-6, real_vs_gen_weight=0.5,
        discriminate_ae_recon=False, discriminate_sample_z=True,
    )
    print('experiment_name: %s' % experiment_name)
    output_dir = os.path.join('out', experiment_name)

    model_path = os.path.join(output_dir, 'arch.pickle')
    print('Loading model from disk')
    print(model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print('Getting data feed')
    model.phase = 'test'
    train_feed, test_feed = dataset.celeba.feeds(
        img_size, batch_size=batch_size, epoch_size=epoch_size,
        split='test', n_augment=0
    )

    save_dir = os.path.join(output_dir, 'User-Testing')
    os.mkdir(save_dir)

    original_x = np.array(test_feed.batches().next()[0])
    samples_z = np.random.normal(size=(len(original_x), n_hidden))
    samples_z = (samples_z).astype(dp.float_)

    print('Saving samples')
    output.samples(model, samples_z, save_dir, img_inverse_transform)

    print('Saving reconstructions')
    output.reconstructions(model, original_x, save_dir, img_inverse_transform)


if __name__ == '__main__':
    run()
