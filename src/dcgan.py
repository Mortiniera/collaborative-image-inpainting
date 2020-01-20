import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
#mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
from dataset import Dataset
import utils as utils
from utils_2 import *
from ops import *

from collaborator import Refiner
from functools import partial

import pickle

class DCGAN(object):
    def __init__(self, sess, flags, dataset):
        self.sess = sess
        self.flags = flags
        self.dataset = Dataset(self.flags, dataset)
        self.image_size = self.dataset.image_size
        self.batch_size = 64

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [1024, 512, 256, 128]  # 4, 8, 16, 32
        self.dis_c = [64, 128, 256, 512]  # 32, 16, 8, 4

        self.rollout_rate = self.flags.rollout_rate
        self.rollout_steps = self.flags.rollout_steps
        self.refiner_batch = self.flags.refiner_batch
        self.dis_shaping_batch = self.flags.dis_shaping_batch
        self.mode = self.flags.mode
        self.observe_evolution = self.flags.observe_evolution

        self._build_net()
        self._tensorboard()
        print("Initialized DCGAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='output')
        self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')

        self.g_samples = self.generator(self.z)
        d_real, d_logit_real = self.discriminator(self.Y, is_reuse=False)
        d_fake, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

        # discriminator loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        self.g_loss_without_mean = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake))

        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        # Optimizer
        self.dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate_adam, beta1=self.flags.beta1)\
            .minimize(self.d_loss, var_list=self.d_vars)
        dis_ops = [self.dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate_adam, beta1=self.flags.beta1)\
            .minimize(self.g_loss, var_list=self.g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)


        #Collaborative Sampling
        self.discriminator_refine = partial(self.discriminator, is_reuse=True)

        def loss_refine(logits):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))

        self.refiner = Refiner(rollout_steps=self.rollout_steps, rollout_rate=self.rollout_rate)
        self.refiner.set_env(self.discriminator_refine, self.feature_to_data, loss_refine)
        input_to_feature = self.input_to_feature(self.z)
        self.g_refine_detem = self.refiner.build_refiner(input_to_feature, self.Y, mode='deterministic',size=self.refiner_batch,
                                                         inpaint=self.mode, evolution=self.observe_evolution, n_channels=self.image_size[2])
        self.g_refine_detem_shape = self.refiner.build_refiner(input_to_feature, self.Y, mode='deterministic', size=self.dis_shaping_batch,
                                                               inpaint=self.mode, evolution=False, n_channels=self.image_size[2])

    def generator(self, data, name='g_', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            data_flatten = flatten(data)

            # 4 x 4
            h0_linear = tf_utils.linear(data_flatten, 4*4*self.gen_c[0], name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, self.gen_c[0]])
            h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
            h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

            # 8 x 8
            h1_deconv = tf_utils.deconv2d(h0_relu, self.gen_c[1], name='h1_deconv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
            h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

            # 16 x 16
            h2_deconv = tf_utils.deconv2d(h1_relu, self.gen_c[2], name='h2_deconv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
            h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

            # 32 x 32
            h3_deconv = tf_utils.deconv2d(h2_relu, self.gen_c[3], name='h3_deconv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
            h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')

            # 64 x 64
            output = tf_utils.deconv2d(h3_relu, self.image_size[2], name='h4_deconv2d')
            return tf.nn.tanh(output)

    def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse) as scope:
            # 64 -> 32
            h0_conv = tf_utils.conv2d(data, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

            # 32 -> 16
            h1_conv = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # 16 -> 8
            h2_conv = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            # 8 -> 4
            h3_conv = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            h3_flatten = flatten(h3_lrelu)
            h4_linear = tf_utils.linear(h3_flatten, 1, name='h4_linear')

            return tf.nn.sigmoid(h4_linear), h4_linear

    def _tensorboard(self):
        # tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def input_to_feature(self, z):
        with tf.variable_scope("g_", reuse=True):
            data_flatten = flatten(z)

            # 4 x 4
            h0_linear = tf_utils.linear(data_flatten, 4 * 4 * self.gen_c[0], name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, self.gen_c[0]])
            h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
            h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

            # 8 x 8
            h1_deconv = tf_utils.deconv2d(h0_relu, self.gen_c[1], name='h1_deconv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
            h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

            # 16 x 16
            h2_deconv = tf_utils.deconv2d(h1_relu, self.gen_c[2], name='h2_deconv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
            h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

            return h2_relu

    def feature_to_data(self, net):
        with tf.variable_scope("g_", reuse=True):
            # 32 x 32
            h3_deconv = tf_utils.deconv2d(net, self.gen_c[3], name='h3_deconv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
            h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')

            # 64 x 64
            output = tf_utils.deconv2d(h3_relu, self.image_size[2], name='h4_deconv2d')
            return tf.nn.tanh(output)

    def train_step(self, imgs):
        feed = {self.z: self.sample_z(num=self.batch_size), self.Y: imgs}

        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed)
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=feed)

        return [d_loss, g_loss], summary

    def sample_imgs(self, fixed_z):
        g_feed = {self.z: self.sample_z(num=self.flags.sample_batch, fixed=True, fixed_z=fixed_z)}
        y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)

        return [y_fakes]

    def sample_z(self, num=64, fixed=False, fixed_z=None):
        if(fixed) :
            return fixed_z
        return np.random.uniform(-1., 1., size=[num, self.flags.z_dim])

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.batch_size),
                                                  ('d_loss', loss[0]), ('g_loss', loss[1]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_batch, *self.image_size))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.04, 0.01
        n_cols, n_rows = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if self.image_size[2] == 3:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                elif self.image_size[2] == 1:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1]), cmap='Greys_r')
                else:
                    raise NotImplementedError

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)

    def discriminator_shaping(self, test_z_list, test_images_list, images_dir, vectors_dir) :
        #Initialize masks
        scale = 0.25
        low, upper = int(self.image_size[0] * scale), int(self.image_size[0] * (1.0 - scale))
        masks = np.ones((self.dis_shaping_batch, 64, 64, 1), dtype=np.float32)
        masks[:, low:upper, low:upper] = 0.

        #Run Discriminator shaping and collaborative sampling alternatively
        refined_samples_list = []
        refined_samples = []

        # Different stages : before DS, after 0.1 epoch, 0.5 epoch, 1 epoch, 1.5 and 2 epoch
        stages = [0, 100, 500, 1000, 1500, 2000]
        epoch = 2
        counter = 0


        # refine samples before discriminator shaping

        for n in range(len(test_z_list)) :
            refined_samples_list.append([])
            refined_samples_list[n].append(self.sess.run(self.g_refine_detem, feed_dict={self.Y: test_images_list[n], self.z: test_z_list[n]}))


        losses = []
        for e in range(epoch) :
            for i in tqdm(range(10)) :
                #Load images and their corresponding previously saved closest encoding latent vector z
                with open("{}/images_{}.pkl".format(images_dir, i), 'rb') as f:
                    training_imgs = pickle.load(f) #100 batch of images
                with open("{}/latent_vectors_{}.pkl".format(vectors_dir, i), 'rb') as f:
                    vectors = pickle.load(f) #100 batch of latent vectors
                for j in range(100) :
                    idx = np.random.choice(100, replace=False) #choose randomly to remove ordering bias
                    context_images = training_imgs[idx] #dis_shaping_batch number of context images
                    latent_vectors = vectors[idx] #dis_shaping_batch number of latent vectors

                    feed = {self.z: latent_vectors, self.Y: context_images}
                    batch_refined = self.sess.run(self.g_refine_detem_shape, feed_dict=feed)[0]

                    if(self.mode == "inpaint") :
                        batch_refined = np.multiply(context_images, masks) + np.multiply(batch_refined, 1. - masks)

                     #shape D network
                    _, d_loss = self.sess.run([self.dis_optim, self.d_loss],
                                             feed_dict={self.g_samples: batch_refined, self.Y: context_images})
                    
                    if(self.flags.print_ds_loss) :
                        losses.append(d_loss)
                        if(j % 10 == 0) :
                            print("--- d_loss : {} ---".format(np.mean(d_loss)))
                            losses = [] #reinitialize window

                    counter += 1
                    if(counter in stages) :
                        print("--- Compute intermediate results : iter_{} ---".format(counter))
                        for n in range(len(test_z_list)):
                            refined_samples_list[n].append(
                                self.sess.run(self.g_refine_detem, feed_dict={self.Y: test_images_list[n], self.z: test_z_list[n]}))

            print("Epoch {}/{}".format(e + 1, epoch))

        return refined_samples_list

    def write(self, filename, ckpt, iteration, psnr_score, frechet_distance, efficiency, calib):
        f = open(filename, "a+")
        f.write("%d    %d    %.4f    %.4f    %.4f    %.4f\r\n"
                % (ckpt, iteration, inception_score, frechet_distance, efficiency, calib))
        f.close()