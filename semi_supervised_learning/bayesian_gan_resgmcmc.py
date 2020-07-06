import os
import sys
import argparse
import json
import time

import numpy as np
from math import ceil

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, Cifar100, SVHN, ImageNet
from bgan_models import BDCGAN

import time


def get_session():
    if tf.get_default_session() is None:
        print "Creating new session"
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print "Using old session"
        _SESSION = tf.get_default_session()

    return _SESSION


def get_gan_labels(lbls):
    # add class 0 which is the "fake" class
    if lbls is not None:
        labels = np.zeros((lbls.shape[0], lbls.shape[1] + 1))
        labels[:, 1:] = lbls
    else:
        labels = None
    return labels


def get_supervised_batches(dataset, size, batch_size, class_ids):

    def batchify_with_size(sampled_imgs, sampled_labels, size):
        rand_idx = np.random.choice(range(sampled_imgs.shape[0]), size, replace=False)
        imgs_ = sampled_imgs[rand_idx]
        lbls_ = sampled_labels[rand_idx]
        rand_idx = np.random.choice(range(imgs_.shape[0]), batch_size, replace=True)
        imgs_ = imgs_[rand_idx]
        lbls_ = lbls_[rand_idx] 
        return imgs_, lbls_

    labeled_image_batches, lblss = [], []
    num_passes = int(ceil(float(size) / batch_size))
    for _ in xrange(num_passes):
        for class_id in class_ids:
            labeled_image_batch, lbls = dataset.next_batch(int(ceil(float(batch_size)/len(class_ids))),
                                                           class_id=class_id)
            labeled_image_batches.append(labeled_image_batch)
            lblss.append(lbls)

    labeled_image_batches = np.concatenate(labeled_image_batches)
    lblss = np.concatenate(lblss)

    if size < batch_size:
        labeled_image_batches, lblss = batchify_with_size(labeled_image_batches, lblss, size)

    shuffle_idx = np.arange(lblss.shape[0]); np.random.shuffle(shuffle_idx)
    labeled_image_batches = labeled_image_batches[shuffle_idx]
    lblss = lblss[shuffle_idx]

    while True:
        i = np.random.randint(max(1, size/batch_size))
        yield (labeled_image_batches[i*batch_size:(i+1)*batch_size],
               lblss[i*batch_size:(i+1)*batch_size])


def get_test_batches(dataset, batch_size):

    try:
        test_imgs, test_lbls = dataset.test_imgs, dataset.test_labels
    except:
        test_imgs, test_lbls = dataset.get_test_set()

    all_test_img_batches, all_test_lbls = [], []
    test_size = test_imgs.shape[0]
    i = 0
    while (i+1)*batch_size <= test_size:
        all_test_img_batches.append(test_imgs[i*batch_size:(i+1)*batch_size])
        all_test_lbls.append(test_lbls[i*batch_size:(i+1)*batch_size])
        i += 1

    return all_test_img_batches, all_test_lbls

def get_test_accuracy(session, dcgan, all_test_img_batches, all_test_lbls):
    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_logits, all_d1_logits, all_s_logits = [], [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_logits, test_d1_logits, test_s_logits = session.run([dcgan.test_D_logits, dcgan.test_D1_logits, dcgan.test_S_logits], feed_dict={dcgan.test_inputs: test_image_batch})
        all_d_logits.append(test_d_logits)
        all_d1_logits.append(test_d1_logits)
        all_s_logits.append(test_s_logits)
        
    
    test_d_logits = np.concatenate(all_d_logits)
    test_d1_logits = np.concatenate(all_d1_logits)
    test_s_logits = np.concatenate(all_s_logits)
    test_lbls = np.concatenate(all_test_lbls)

    not_fake = np.where(np.argmax(test_d_logits, 1) > 0)[0]
    not_fake1 = np.where(np.argmax(test_d1_logits, 1) > 0)[0]
    if len(not_fake) < 1000:
        print "WARNING: not enough samples for SS results"
        return -1, -1, -1
    if len(not_fake1) < 1000:
        print "WARNING: not enough samples for SS results"
        return -1, -1, -1
    semi_sup_acc = (100. * np.sum(np.argmax(test_d_logits[not_fake], 1) == np.argmax(test_lbls[not_fake], 1) + 1))\
                   / len(not_fake)
    semi_sup1_acc = (100. * np.sum(np.argmax(test_d1_logits[not_fake1], 1) == np.argmax(test_lbls[not_fake1], 1) + 1))\
                   / len(not_fake1)
    sup_acc = (100. * np.sum(np.argmax(test_s_logits, 1) == np.argmax(test_lbls, 1)))\
              / test_lbls.shape[0]
    return sup_acc, semi_sup_acc, semi_sup1_acc


def get_test_variance(session, dcgan, dataset, batch_size, z_dim, fileNameV):
    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    d_losses, d1_losses = [], []
    
    if hasattr(dataset, "supervised_batches"):
        # implement own data feeder if data doesnt fit in memory
        supervised_batches = dataset.supervised_batches(args.N, batch_size)
    else:
        supervised_batches = get_supervised_batches(dataset, args.N, batch_size, range(dataset.num_classes))
    
    ENOUGH_INTR_NUM = 1000000
    for i in range(ENOUGH_INTR_NUM):
        labeled_image_batch, labels = supervised_batches.next()
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        image_batch, _ = dataset.next_batch(batch_size, class_id=None)
        
        
        
        d_loss, d1_loss = session.run([dcgan.d_loss_semi, dcgan.d1_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch, dcgan.labels: get_gan_labels(labels), dcgan.inputs: image_batch, dcgan.z: batch_z})
        
        
        
        d_losses.append(d_loss)
        d1_losses.append(d1_loss)
        if len(d_losses) > args.repeats:
            break
        
    with open(fileNameV, 'a') as f_variance:
        f_variance.write("%.2f %.2f \n" % (float(np.std(d_losses)), float(np.std(d1_losses))))
    return float(np.std(d_losses)), float(np.std(d1_losses))

def b_dcgan(dataset, dataset_get_variance, args):
    fileNameV = "variance"
    fileNameAcc = "accuracy"
    if args.N == 1000: 
        fileNameV += "1000"
        fileNameAcc += "1000"
    if args.N == 2000: 
        fileNameV += "2000"
        fileNameAcc += "2000"
    if args.N == 4000: 
        fileNameV += "4000"
        fileNameAcc += "4000"
    if args.N == 8000: 
        fileNameV += "8000"
        fileNameAcc += "8000"
    fileNameV += args.fileName
    fileNameAcc += args.fileName
    fileNameV += ".txt"
    fileNameAcc += ".txt"
    f_variance = open(fileNameV, "wb")
    f_variance.write("Low, High Variance here: \n")
    f_accuracy = open(fileNameAcc, "wb")
    f_variance.close()
    f_accuracy.close()
    corrections = [[], []]
    mv_corrections = []
    mv_corrections.append(sys.float_info.max)
    mv_corrections.append(sys.float_info.max)
    
    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size
    dataset_size = dataset.dataset_size

    session = get_session()
    if args.random_seed is not None:
	tf.set_random_seed(args.random_seed)
    # due to how much the TF code sucks all functions take fixed batch_size at all times
    dcgan = BDCGAN(x_dim, z_dim, dataset_size, batch_size=batch_size, J=args.J, M=args.M, 
                   lr=args.lr, optimizer=args.optimizer, gen_observed=args.gen_observed,
                   num_classes=dataset.num_classes if args.semi_supervised else 1)
    dcgan.set_parallel_chain_params(args.invT, args.Tgap, args.LRgap, args.Egap, args.anneal, args.lr_anneal)
    
    print "Starting session"
    session.run(tf.global_variables_initializer())

    print "Starting training loop"
        
    num_train_iter = args.train_iter

    if hasattr(dataset, "supervised_batches"):
        # implement own data feeder if data doesnt fit in memory
        supervised_batches = dataset.supervised_batches(args.N, batch_size)
    else:
        supervised_batches = get_supervised_batches(dataset, args.N, batch_size, range(dataset.num_classes))

    test_image_batches, test_label_batches = get_test_batches(dataset, batch_size)

    optimizer_dict = {"semi_d": dcgan.d_optim_semi_adam,
                      "semi_d1": dcgan.d1_optim_semi_adam,
                      "sup_d": dcgan.s_optim_adam,
                      "gen": dcgan.g_optims_adam,
                      "gen1": dcgan.g1_optims_adam}

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    base_learning_rate1 = args.lr / args.LRgap # for now we use same learning rate for Ds and Gs
    lr_decay_rate = 3.0 # args.lr_decay
    zero_lr = 0.0
    swap_count = 0

    for train_iter in range(num_train_iter):
        if train_iter == 5000:
            print "Switching to user-specified optimizer"
            optimizer_dict = {"semi_d": dcgan.d_optim_semi,
                              "semi_d1": dcgan.d1_optim_semi,
                              "sup_d": dcgan.s_optim,
                              "gen": dcgan.g_optims,
                              "gen1": dcgan.g1_optims}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))
        
        learning_rate1 = base_learning_rate1 * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))

        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        image_batch, _ = dataset.next_batch(batch_size, class_id=None)
        
        if args.semi_supervised:
            labeled_image_batch, labels = supervised_batches.next()
           
            _, d_loss = session.run([optimizer_dict["semi_d"], dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch, 
                                                                                              dcgan.labels: get_gan_labels(labels),
                                                                                              dcgan.inputs: image_batch,
                                                                                              dcgan.z: batch_z,
                                                                                              dcgan.d_semi_learning_rate: learning_rate})
            
            _, d1_loss = session.run([optimizer_dict["semi_d1"], dcgan.d1_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch,
                                                                                              dcgan.labels: get_gan_labels(labels),
                                                                                              dcgan.inputs: image_batch,
                                                                                              dcgan.z: batch_z,
                                                                                              dcgan.d1_semi_learning_rate: learning_rate1})
            
            _, s_loss = session.run([optimizer_dict["sup_d"], dcgan.s_loss], feed_dict={dcgan.inputs: labeled_image_batch,
                                                                                        dcgan.lbls: labels})

        bias = (mv_corrections[0] + mv_corrections[1]) * args.bias_multi

        if np.log(np.random.uniform(0, 1)) < (d1_loss - d_loss + bias) * (args.invT - args.invT*args.Tgap) and args.baseline == 0:
            
            swap_count += 1
            print "Copy Iter %i, Copy Count %i" % (train_iter, swap_count)
            print "Disc1 (high temperature) loss = %.2f, Gen loss = %s" % (d1_loss, ", ".join(["%.2f" % gl for gl in g1_losses]))
            print "Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % g for g in g_losses]))
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Copy Iter %i, Copy Count %i \n" % (train_iter, swap_count))
                f_accuracy.write("Disc1 (high temperature) loss = %.2f, Gen loss = %s \n" % (d1_loss, ", ".join(["%.2f" % gl for gl in g1_losses])))
                f_accuracy.write("Disc loss = %.2f, Gen loss = %s \n" % (d_loss, ", ".join(["%.2f" % g for g in g_losses])))
                        
            print("Copy status of the second discriminator to the first one")
            s_acc, ss_acc, ss1_acc= get_test_accuracy(session, dcgan, test_image_batches, test_label_batches)
            print "Semi-sup classification acc before copy: %.2f" % (ss_acc)
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Copy status of the second discriminator to the first one \n")
                f_accuracy.write("Semi-sup classification acc before copy: %.2f \n" % (ss_acc))
            dcgan.copy_discriminator(session)
            
            # get test set performance on real labels only for both GAN-based classifier and standard one
            s_acc, ss_acc, ss1_acc= get_test_accuracy(session, dcgan, test_image_batches, test_label_batches)
            print "Semi-sup 1 (high temperature) classification acc before copy: %.2f" % (ss1_acc)
            print "Semi-sup classification acc after copy: %.2f" % (ss_acc)
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Semi-sup 1 (high temperature) classification acc before copy: %.2f \n" % (ss1_acc))
                f_accuracy.write("Semi-sup classification acc after copy: %.2f \n" % (ss_acc))
        g_losses = []
        g1_losses = []
        for gi in xrange(dcgan.num_gen):
            # compute g_sample loss
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
            for m in range(dcgan.num_mcmc):
                _, g_loss = session.run([optimizer_dict["gen"][gi*dcgan.num_mcmc+m], dcgan.generation["g_losses"][gi*dcgan.num_mcmc+m]],
                                        feed_dict={dcgan.z: batch_z, dcgan.g_learning_rate: learning_rate})
                
                _, g1_loss = session.run([optimizer_dict["gen1"][gi*dcgan.num_mcmc+m], dcgan.generation["g1_losses"][gi*dcgan.num_mcmc+m]],
                                        feed_dict={dcgan.z: batch_z, dcgan.g_learning_rate: learning_rate1})
                g_losses.append(g_loss)
                g1_losses.append(g1_loss)
        
                

        if train_iter > 0 and train_iter % args.n_save == 0:
            print "Iter %i" % train_iter
            print "Disc1 (high temperature) loss = %.2f, Gen loss = %s" % (d1_loss, ", ".join(["%.2f" % gl for gl in g1_losses]))
            # print "Disc1 (high temperature) loss = %.2f" % (d1_loss)
            print "Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % g for g in g_losses]))
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Iter %i \n" % train_iter)
                f_accuracy.write("Disc1 (high temperature) loss = %.2f, Gen loss = %s \n" % (d1_loss, ", ".join(["%.2f" % gl for gl in g1_losses])))
                f_accuracy.write("Disc loss = %.2f, Gen loss = %s \n" % (d_loss, ", ".join(["%.2f" % g for g in g_losses])))
            
            # collect samples
            if args.save_samples: # saving samples
                all_sampled_imgs = []
                for gi in xrange(dcgan.num_gen):
                    _imgs, _ps = [], []
                    for _ in range(10):
                        sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                        sampled_imgs, sampled_probs = session.run([dcgan.generation["gen_samplers"][gi*dcgan.num_mcmc],
                                                                   dcgan.generation["d_probs"][gi*dcgan.num_mcmc]],
                                                                  feed_dict={dcgan.z: sample_z})
                        _imgs.append(sampled_imgs)
                        _ps.append(sampled_probs)

                    sampled_imgs = np.concatenate(_imgs); sampled_probs = np.concatenate(_ps)
                    all_sampled_imgs.append([sampled_imgs, sampled_probs[:, 1:].sum(1)])
                    
            s_acc, ss_acc, ss1_acc= get_test_accuracy(session, dcgan, test_image_batches, test_label_batches)
            d_std, d1_std = get_test_variance(session, dcgan, dataset_get_variance, batch_size, z_dim, fileNameV)
            print "Semi-sup 1 (high temperature) classification before correction acc: %.2f" % (ss1_acc)
            print "Semi-sup classification acc: %.2f" % (ss_acc)
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Semi-sup 1 (high temperature) classification before correction acc: %.2f \n" % (ss1_acc))
                f_accuracy.write("Semi-sup classification acc: %.2f \n" % (ss_acc))
                f_accuracy.write("============================================ \n")
            
            # moving window average
            corrections[0].append(0.5 * d_std**2)
            corrections[1].append(0.5 * d1_std**2)
            # exponential smoothing average
            if mv_corrections[0] == sys.float_info.max:
                mv_corrections[0] = 0.5 * d_std**2
            else:
                mv_corrections[0] = (1 - args.alpha) * mv_corrections[0] + args.alpha * 0.5 * d_std ** 2
            if mv_corrections[1] == sys.float_info.max:
                mv_corrections[1] = 0.5 * d1_std**2
            else:
                mv_corrections[1] = (1 - args.alpha) * mv_corrections[1] + args.alpha * 0.5 * d1_std ** 2
            
            
            # print "Sup classification acc: %.2f" % (s_acc)
            print "Semi-sup 1 (high temperature) classification acc: %.2f" % (ss1_acc)
            print "Semi-sup classification acc: %.2f" % (ss_acc)
            with open(fileNameAcc, 'a') as f_accuracy:
                f_accuracy.write("Semi-sup 1 (high temperature) classification acc: %.2f \n" % (ss1_acc))
                f_accuracy.write("Semi-sup classification acc: %.2f \n" % (ss_acc))
                f_accuracy.write("============================================ \n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")
    
    parser.add_argument('--fileName',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")

    parser.add_argument('--n_save',
                        type=int,
                        default=100,
                        help="every n_save iteration save samples and weights")
    
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    
    parser.add_argument('--gen_observed',
                        type=int,
                        default=1000,
                        help='number of data "observed" by generator')

    parser.add_argument('--data_path',
                        type=str,
                        default='./datasets/',
                        help='path to where the datasets live')

    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='datasate name mnist etc.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="minibatch size")

    parser.add_argument('--prior_std',
                        type=float,
                        default=1.0,
                        help="NN weight prior std.")

    parser.add_argument('--numz',
                        type=int,
                        dest="J",
                        default=1,
                        help="number of samples of z to integrate it out")

    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=1,
                        help="number of MCMC NN weight samples per z")

    parser.add_argument('--N',
                        type=int,
                        default=128,
                        help="number of supervised data samples")

    parser.add_argument('--semi_supervised',
                        action="store_true",
                        help="do semi-supervised learning")

    parser.add_argument('--train_iter',
                        type=int,
                        default=50000,
                        help="number of training iterations")

    parser.add_argument('--wasserstein',
                        action="store_true",
                        help="wasserstein GAN")

    parser.add_argument('--ml_ensemble',
                        type=int,
                        default=0,
                        help="if specified, an ensemble of --ml_ensemble ML DCGANs is trained")

    parser.add_argument('--save_samples',
                        action="store_true",
                        help="wether to save generated samples")
    
    parser.add_argument('--save_weights',
                        action="store_true",
                        help="wether to save weights")

    parser.add_argument('--random_seed',
                        type=int,
                        default=None,
                        help="random seed")
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.003,
                        help="learning rate")

    parser.add_argument('--lr_decay',
                        type=float,
                        default=1.003,
                        help="learning rate")

    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help="optimizer --- 'adam' or 'sgd'")
    parser.add_argument('--gpu',
                        type=str,
                        default="0",
                        help="GPU number")
    
    # Parallel chain hyperparameters
    parser.add_argument('-chains',
                        default=2,
                        type=int,
                        help='Total number of chains')
    parser.add_argument('-types',
                        default='greedy',
                        type=str,
                        help='swap type: greedy (low T copy high T), swap (low high T swap)')
    parser.add_argument('-invT',
                        default=1,
                        type=float,
                        help='Inverse temperature for high temperature chain')
    parser.add_argument('-Tgap',
                        default=1,
                        type=float,
                        help='Temperature gap between chains')
    parser.add_argument('-LRgap',
                        default=1,
                        type=float,
                        help='Learning rate gap between chains')
    parser.add_argument('-Egap',
                        default=1.025,
                        type=float,
                        help='Energy gap between partitions')
    parser.add_argument('-anneal',
                        default=1.0,
                        type=float,
                        help='simulated annealing factor')
    parser.add_argument('-lr_anneal',
                        default=0.992,
                        type=float,
                        help='lr simulated annealing factor')
    parser.add_argument('-bias_multi',
                        default=5.0,
                        type=float,
                        help='multiplier for bias')
    parser.add_argument('-alpha',
                        default=0.1,
                        type=float,
                        help='Constant for exponential smoothness')
    parser.add_argument('-repeats',
                        default=60,
                        type=float,
                        help='Number of samples to estimate sample std')
    parser.add_argument('-baseline',
                        default=0,
                        type=int,
                        help='Baseline (1) and Two Chain (0)')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.random_seed is not None:
        #np.random.seed(args.random_seed)
        np.random.seed(2222)
        tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print "Creating %s" % args.out_dir
        os.makedirs(args.out_dir)
    args.out_dir = os.path.join(args.out_dir, "bgan_%s_%i" % (args.dataset, int(time.time())))
    os.makedirs(args.out_dir)

    import pprint
    with open(os.path.join(args.out_dir, "hypers.txt"), "w") as hf:
        hf.write("Hyper settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
        
    celeb_path = os.path.join(args.data_path, "celebA")
    cifar_path = os.path.join(args.data_path, "cifar10")
    cifar100_path = os.path.join(args.data_path, "cifar100")
    svhn_path = os.path.join(args.data_path, "svhn")
    mnist_path = os.path.join(args.data_path, "mnist") # can leave empty, data will self-populate
    imagenet_path = os.path.join(args.data_path, args.dataset)

    if args.dataset == "mnist":
        dataset = MnistDataset(mnist_path)
    elif args.dataset == "celeb":
        dataset = CelebDataset(celeb_path)
    elif args.dataset == "cifar100":
        dataset = Cifar100(cifar100_path)
        dataset_get_variance = Cifar100(cifar100_path)
    elif args.dataset == "cifar":
        dataset = Cifar10(cifar_path)
        dataset_get_variance = Cifar10(cifar_path)
    
    elif args.dataset == "svhn":
        dataset = SVHN(svhn_path)
        dataset_get_variance = SVHN(svhn_path)
    elif "imagenet" in args.dataset:
        num_classes = int(args.dataset.split("_")[-1])
        dataset = ImageNet(imagenet_path, num_classes)
    else:
        raise RuntimeError("invalid dataset %s" % args.dataset)
        
    ### main call
    b_dcgan(dataset, dataset_get_variance, args)
