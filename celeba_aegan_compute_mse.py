#!/usr/bin/env python

import os
import pickle
import numpy as np
import time

import dataset.celeba
import aegan


def run():
    experiment_name = 'celeba'

    img_size = 64
    epoch_size = 1
    batch_size = 64

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

    save_dir = os.path.join(output_dir, 'MSE-Results')
    if not os.path.expanduser(save_dir):
        os.mkdir(save_dir)

    t = time.time()

    # Data in [-1,1] range
    mse_train = 0.0
    j = 0
    # This loop only iterates once
    for batch in train_feed.batches():
        original_x = np.array(batch[0])

        recon_x = model.decode(model.encode(original_x))

        mse_train += np.sum(np.mean((original_x - recon_x) ** 2, axis=(1, 2, 3)))
        j += 1
    mse_train /= batch_size * j

    mse_test = 0.0
    j = 0
    for batch in test_feed.batches():
        original_x = np.array(batch[0])

        recon_x = model.decode(model.encode(original_x))

        mse_test += np.sum(np.mean((original_x - recon_x) ** 2, axis=(1, 2, 3)))
        j += 1
    mse_test /= batch_size * j

    total_time = time.time() - t
    print("MSE train: {}, MSE test: {}, time: {} minutes".format(mse_train, mse_test, total_time / (60)))
    # ((x + 1)/2 - (y + 1)/2)^2 = (x/2 - y/2)^2 = ((x - y)*1/2)^2 = [(x - y)^2]*[(1/2)^2] = (1/4)*(x-y)**2
    print("MSE for [0,1]: {}, num batches: {}, num samples: {}".format(mse_test/4.0, j, test_feed.n_samples))
    np.save(os.path.join(save_dir, 'mse.npy'), [mse_train, mse_test, total_time])

if __name__ == '__main__':
    run()
