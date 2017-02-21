#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy as sp
import deeppy as dp

import dataset.celeba
from model import ae
from dataset.util import img_inverse_transform
from video import Video
import output

import architectures

def run():
    experiment_name = 'celeba-ae'

    img_size = 64
    epoch_size = 250
    batch_size = 64
    n_hidden = 128
    n_augment = int(6e5)
    train_feed, test_feed = dataset.celeba.feeds(
        img_size, split='test', batch_size=batch_size, epoch_size=epoch_size,
        n_augment=n_augment,
    )

    experiment_name += '_nhidden%i' % n_hidden

    out_dir = os.path.join('out', experiment_name)
    arch_path = os.path.join(out_dir, 'arch.pickle')
    start_arch_path = arch_path
    start_arch_path = None

    print('experiment_name', experiment_name)
    print('start_arch_path', start_arch_path)
    print('arch_path', arch_path)

    ae_kind = 'variational'
    # Setup network
    if start_arch_path is None:
        print('Creating new model')
        encoder, decoder, _ = architectures.img64x64()
        if ae_kind == 'variational':
            latent_encoder = architectures.vae_latent_encoder(n_hidden)
        elif ae_kind == 'adversarial':
            latent_encoder = architectures.aae_latent_encoder(n_hidden)
    else:
        print('Starting from %s' % start_arch_path)
        with open(start_arch_path, 'rb') as f:
            encoder, latent_encoder, decoder = pickle.load(f)

    model = ae.Autoencoder(
        encoder=encoder,
        latent_encoder=latent_encoder,
        decoder=decoder,
    )
    model.recon_error = ae.NLLNormal()

    # Plotting
    n_examples = 64
    original_x, = test_feed.batches().next()
    original_x = np.array(original_x)[:n_examples]
    samples_z = np.random.normal(size=(n_examples, n_hidden))
    samples_z = (samples_z).astype(dp.float_)

    n_epochs = 250
    lr_start = 0.025
    lr_stop = 0.00001
    lr_gamma = 0.75
    # Train network
    learn_rule = dp.RMSProp()
    trainer = dp.GradientDescent(model, train_feed, learn_rule)
    annealer = dp.GammaAnnealer(lr_start, lr_stop, n_epochs, gamma=lr_gamma)
    try:
        recon_video = Video(os.path.join(out_dir, 'convergence_recon.mp4'))
        sample_video = Video(os.path.join(out_dir, 'convergence_samples.mp4'))
        sp.misc.imsave(os.path.join(out_dir, 'examples.png'),
                       dp.misc.img_tile(img_inverse_transform(original_x)))
        for e in range(n_epochs):
            model.phase = 'train'
            model.setup(*train_feed.shapes)
            learn_rule.learn_rate = annealer.value(e) / batch_size
            loss = trainer.train_epoch()

            model.phase = 'test'
            original_z = model.encode(original_x)
            recon_x = model.decode(original_z)
            samples_x = model.decode(samples_z)
            recon_x = img_inverse_transform(recon_x)
            samples_x = img_inverse_transform(model.decode(samples_z))
            recon_video.append(dp.misc.img_tile(recon_x))
            sample_video.append(dp.misc.img_tile(samples_x))
            likelihood = model.likelihood(test_feed)
            print('epoch %i   Train loss:%.4f  Test likelihood:%.4f  Learn Rate:%.4f' %
                  (e, np.mean(loss), np.mean(likelihood), learn_rule.learn_rate))
    except KeyboardInterrupt:
        pass
    print('Saving model to disk')
    with open(arch_path, 'wb') as f:
        pickle.dump((encoder, latent_encoder, decoder), f)

    model.phase = 'test'
    n_examples = 100
    test_feed.reset()
    original_x = np.array(test_feed.batches().next()[0])[:n_examples]
    samples_z = np.random.normal(size=(n_examples, n_hidden))
    output.samples(model, samples_z, out_dir, img_inverse_transform)
    output.reconstructions(model, original_x, out_dir,
                           img_inverse_transform)
    original_z = model.encode(original_x)
    output.walk(model, original_z, out_dir, img_inverse_transform)

if __name__ == '__main__':
    run()

