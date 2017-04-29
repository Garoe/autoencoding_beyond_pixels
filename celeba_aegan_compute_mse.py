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

    c = -0.5 * np.log(2 * np.pi)
    sigma = 1.0
    multiplier = 1.0 / (2.0 * sigma ** 2)

    nll_test = 0.0
    mse_test = 0.0
    j = 0
    for batch in test_feed.batches():
        original_x = np.array(batch[0])

        recon_x = model.decode(model.encode(original_x))

        mse_test += np.sum(np.mean((original_x - recon_x) ** 2, axis=(1, 2, 3)))

        # c - multiplier*(pred - target)**2
        tmp = original_x - recon_x
        tmp **= 2.0
        tmp *= -multiplier
        tmp += c

        # axis = tuple(range(1, len(original_x.shape)))
        # nll_test += np.sum(tmp, axis=axis)
        nll_test += np.sum(tmp)

        j += 1
    mse_test /= batch_size * j
    nll_test /= batch_size * j

    total_time = time.time() - t

    def print_write(file, line):
        print(line)
        file.write(line + "\n")

    with open(os.path.join(save_dir, 'mse.txt'), "w") as file:
        line = "Num batches: {}, num samples: {}, time: {} minutes".format(j, test_feed.n_samples, total_time / 60.0)
        print_write(file, line)

        line = "MSE train: {}, MSE test: {}".format(mse_train, mse_test)
        print_write(file, line)

        line = "NLL test: {}".format(nll_test)
        print_write(file, line)

        # ((x + 1)/2 - (y + 1)/2)^2 = (x/2 - y/2)^2 = ((x - y)*1/2)^2 = [(x - y)^2]*[(1/2)^2] = (1/4)*(x-y)^2
        line = "MSE for [0,1]: {}".format(mse_test / 4.0)
        print_write(file, line)

        # ((x*255 - y * 255)^2 = (255^2) * (x-y)^2
        line = "MSE for [0,255]: {}".format(mse_test * (255 ** 2))
        print_write(file, line)

        line = "L2 error for [0,1]: {}".format(np.sqrt(mse_test / 4.0))
        print_write(file, line)

    np.save(os.path.join(save_dir, 'mse.npy'), [mse_train, mse_test, nll_test, total_time])


if __name__ == '__main__':
    run()
