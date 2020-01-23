import os
import time
import numpy as np
import tensorflow as tf

from dataset import Dataset
from inpaint_model import ModelInpaint
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pickle
import cv2

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray



class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags, self.flags.dataset)
        self.model = ModelInpaint(self.sess, self.flags, self.flags.dataset)

        self._make_folders()
        self.iter_time = 0

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.test_results = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
        self.images_dir = "{}/images".format(self.flags.dataset)
        self.vectors_dir = "{}/vectors/{}".format(self.flags.dataset, self.flags.earlier_stage)

        if not os.path.isdir(self.test_results):
            os.makedirs(self.test_results)
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.isdir(self.vectors_dir):
            os.makedirs(self.vectors_dir)

        self.evolution_dir  = ""

        self.test_out_dir = "{}/inpaint/{}/{}".format(self.flags.dataset, self.flags.load_model, self.flags.mode)

        for n in range(self.flags.num_try) :
            if not os.path.isdir(self.test_out_dir  + "/{}/".format(n)):
                os.makedirs(self.test_out_dir  + "/{}/".format(n))

        if (self.flags.observe_evolution):
            self.evolution_dir = "{}/evolution".format(self.test_out_dir)
        if(self.flags.mode not in ["standard", "inpaint"]) :
            print("Choose mode between [standard | inpaint ]")
            raise NotImplementedError

        for n in range(self.flags.num_try) :
            if not os.path.isdir(self.evolution_dir + "/{}/".format(n)):
                os.makedirs(self.evolution_dir + "/{}/".format(n))


        self.train_writer = tf.summary.FileWriter("{}/inpaint/{}/{}/log".format(
            self.flags.dataset, self.flags.load_model, self.flags.mask_type),
            graph_def=self.sess.graph_def)

    def find_closest_encoding(self, load=False, index=0, prev=False, for_discriminator_shaping=False, images=None) :
        if(load) :
            if self.load_model():
                print(' [*] Load SUCCESS!')
        else:
            print("Already loaded")

        if(prev) :
            print(' [!] Already loaded !')
        
        #Select batch of context images from the validation set
        context_images = self.dataset.val_next_batch(batch_size=self.flags.sample_batch)

        if(for_discriminator_shaping) :
            context_images = images

        #Initialize model
        self.model.preprocess()
        best_loss = np.ones(self.flags.sample_batch) * 1e10
        best_outs = np.zeros_like(context_images)
        best_zs = [0.0] * self.flags.sample_batch

        for iter_time in range(self.flags.iters):
            batch_z, loss, img_outs, summary = self.model(context_images, iter_time)  # inference

            # save best generated results according to the total loss
            for iter_loss in range(self.flags.sample_batch):
                if best_loss[iter_loss] > loss[2][iter_loss]:  # total loss
                    best_loss[iter_loss] = loss[2][iter_loss]
                    best_outs[iter_loss] = img_outs[iter_loss]
                    best_zs[iter_loss] = batch_z[iter_loss]

            self.model.print_info(loss, iter_time)  # print loss information

        # Save the best latent vectors and corresponding context images
        latent_vectors = np.array(best_zs)

        return latent_vectors, context_images


    def luminance(self, img1, img2, n_channels) :
        # Calculate the maximum error for each pixel accross each channel
        if(n_channels == 3) :
            error_r = np.fabs(np.subtract(img2[:, :,:,0], img1[:, :,:,0]))
            error_g = np.fabs(np.subtract(img2[:, :,:,1], img1[:, :,:,1]))
            error_b = np.fabs(np.subtract(img2[:, :,:,2], img1[:, :,:,2]))

            lum_img = np.max(np.max(error_r, error_g, axis=1), error_b, axis=1)
        else :
            lum_img = np.fabs(np.subtract(img2[:, :, :], img1[:, :, :]))

        #turn colors upside down -> make the larger differences "brighter"
        lum_img = np.negative(lum_img)
        lum_img = np.reshape(lum_img, (self.flags.sample_batch, self.flags.img_size, self.flags.img_size, n_channels))
        return lum_img

    def plot_inpainted_results(self, imgs, indice, directory, heatmap_row = -1, inpaint=False) :

        #mask parameters
        scale = 0.25
        low, upper = int(self.flags.img_size * scale), int(self.flags.img_size * (1.0 - scale))

        # parameters for plot size
        scale, margin = 0.04, 0.01
        n_cols, n_rows = self.flags.sample_batch, int(len(imgs) / self.flags.sample_batch)
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                current_image = imgs[row_index * n_cols + col_index]

                if(row_index == heatmap_row) : #Heatmap row
                    plt.imshow(current_image.reshape(
                        self.flags.img_size, self.flags.img_size), cmap='Spectral')
                elif(row_index == 0) : #Corrupted image
                    current_image = (current_image + 1 ) / 2.
                    current_image[low:upper, low:upper] = 0.
                    if(self.dataset.image_size[2] == 3) :
                        plt.imshow(current_image.reshape(
                          self.flags.img_size, self.flags.img_size, self.dataset.image_size[2]), cmap='Greys_r')
                    else :
                        plt.imshow(current_image.reshape(
                            self.flags.img_size, self.flags.img_size), cmap='Greys_r')
                else :
                    current_image = (current_image + 1) / 2.
                    if(self.dataset.image_size[2] == 3) :
                        plt.imshow(current_image.reshape(
                          self.flags.img_size, self.flags.img_size, self.dataset.image_size[2]), cmap='Greys_r')
                    else :
                        plt.imshow(current_image.reshape(
                            self.flags.img_size, self.flags.img_size), cmap='Greys_r')

                #Save figure
                plt.savefig("{}/image_{}.png".format(directory, indice), bbox_inches='tight')
    
    
    # scale an array of images to a new size
    def scale_images(self, images, new_shape):
      images_list = list()
      for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
      return asarray(images_list)
    
    # assumes images have any shape and pixels in [0,255]
    def calculate_inception_score(self, images, n_split=10, eps=1E-16):
      # load inception v3 model
      model = InceptionV3()
      # enumerate splits of images/predictions
      scores = list()
      n_part = floor(images.shape[0] / n_split)
      for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = self.scale_images(subset, (299,299,3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
      # average across images
      is_avg, is_std = mean(scores), std(scores)
      return is_avg, is_std
    
    
    def collaborative_sampling_inpainting(self) :
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')


        #For metrics computings
        original_images = []
        original_method_images = []
        new_method_images = []

        context_images_list = []
        latent_vectors_list = []
        gen_samples_list = []
        inpaint_gen_samples_list = []



        for n in range(self.flags.num_try) :

            latent_vectors, context_images = self.find_closest_encoding(index=self.flags.test_number)
            #Sanity check
            assert latent_vectors.shape[0] == context_images.shape[0], "Not same number of vectors and images to test !"
            assert self.flags.sample_batch == context_images.shape[0], "Sample batch different than number of images to test !"


            


            print("------- Generate reconstructed image ---------")
            # Obtain generated results directly from the generator output,i.e, without any refinement
            gen_samples = self.model.dcgan.sample_imgs(fixed_z=latent_vectors)[0]

            context_images_list.append(context_images)
            latent_vectors_list.append(latent_vectors)
            gen_samples_list.append(gen_samples)
            
            self.model.preprocess()
            print("--------MODE : {} ------".format(self.flags.mode))

            
            if (self.flags.mode == "inpaint"):  # Surround the generated outputs with their context
                inpaint_gen_samples = np.multiply(context_images, self.model.masks) \
                                      + np.multiply(gen_samples, 1. - self.model.masks)
                inpaint_gen_samples_list.append(inpaint_gen_samples)
            


        
        
        print("-------- START COLLABORATIVE SAMPLING AND DISCRIMINATOR SHAPING ------")
        #Include Collaborative Sampling here
        all_results = self.model.dcgan.discriminator_shaping(latent_vectors_list, context_images_list, self.images_dir,
                                                        self.vectors_dir)


        for n in tqdm(range(len(all_results))) :
            results = all_results[n]

            for i in range(len(results)) : #for all stages
                cs_samples = results[i]

                if (self.flags.observe_evolution):
                    self.evolution_epoch_dir = "{}/{}/stage_{}".format(self.evolution_dir, n, i)

                    if not os.path.isdir(self.evolution_epoch_dir):
                        os.makedirs(self.evolution_epoch_dir)



                for forward_step in range(len(cs_samples)):
                    refined_samples = cs_samples[forward_step]

                    inpaint_refined_samples = []
                    if (self.flags.mode == "inpaint"):  # Surround the generated outputs with their context
                        inpaint_refined_samples = np.multiply(context_images_list[n], self.model.masks) + np.multiply(refined_samples, 1. - self.model.masks)

                        heatmap_row = -1
                        if (self.flags.heatmap):
                            lum_imgs = self.luminance(img1=inpaint_gen_samples_list[n], img2=inpaint_refined_samples,
                                                      n_channels=self.dataset.image_size[2])

                        all_imgs = []
                        for img in context_images_list[n]:  # masked images
                            all_imgs.append(img)
                            if((i == len(results) -1) and (forward_step == len(cs_samples) -1)) :
                                original_images.append(img)
                        for img in inpaint_gen_samples_list[n]:  # output of the generator surrounded by context
                            all_imgs.append(img)
                            if((i == len(results) -1) and (forward_step == len(cs_samples) -1)) :
                                original_method_images.append(img)
                        if (self.flags.heatmap):
                            for img in lum_imgs:  # heatmap between inpaint refined samples and inpaint output of the generator
                                all_imgs.append(img)
                            heatmap_row = 2
                        for img in inpaint_refined_samples:  # refined samples after discriminator shaping surrounded by context
                            all_imgs.append(img)
                            if((i == len(results) -1) and (forward_step == len(cs_samples) -1)) :
                                new_method_images.append(img)
                        for img in context_images_list[n]:  # original images
                            all_imgs.append(img)

                        # Plot and save results
                        if (self.flags.observe_evolution):
                            self.plot_inpainted_results(all_imgs, forward_step, directory=self.evolution_epoch_dir, heatmap_row=heatmap_row)

                    else :
                        heatmap_row = -1
                        if (self.flags.heatmap):
                            lum_imgs = self.luminance(img1=gen_samples_list[n], img2=refined_samples,
                                                      n_channels=self.dataset.image_size[2])

                        all_imgs = []
                        for img in context_images_list[n]:  # masked images
                            all_imgs.append(img)
                        for img in gen_samples_list[n]:  # output of the generator
                            all_imgs.append(img)
                        if (self.flags.heatmap):
                            for img in lum_imgs:  # heatmap between refined samples and output of the generator
                                all_imgs.append(img)
                            heatmap_row = 2
                        for img in refined_samples:  # refined samples after discriminator shaping
                            all_imgs.append(img)
                        for img in context_images_list[n]:  # original images
                            all_imgs.append(img)

                        #Plot and save results
                        if(self.flags.observe_evolution) :
                            self.plot_inpainted_results(all_imgs, forward_step, directory=self.evolution_epoch_dir, heatmap_row=heatmap_row)
                if(not self.flags.eval_mode) :
                    self.plot_inpainted_results(all_imgs, i, directory=self.test_out_dir + "/{}".format(n), heatmap_row=heatmap_row)

        print("------------ FINISHED ! ALL PLOTS SAVED ---------------------------")
        
        if(self.flags.mode == "inpaint") :
            #f = open(self.test_out_dir + "/metrics.txt","a+")
            
            p1 = 0
            p2 = 0
            s1 = 0
            s2 = 0
            for k in tqdm(range(len(original_images))) :
                p1 += compare_psnr(original_images[k], original_method_images[k], data_range=2)
                p2 += compare_psnr(original_images[k], new_method_images[k], data_range=2)
                s1 += compare_ssim(original_images[k], original_method_images[k], data_range=2, multichannel=True)
                s2 += compare_ssim(original_images[k], new_method_images[k], data_range=2, multichannel=True)

            print("Semantic Image Inpainting PSNR : {}".format(p1/len(original_images)))
            print("Collaborative Image Inpainting PSNR : {}".format(p2/len(original_images)))
            print("Semantic Image Inpainting SSIM : {}".format(s1/len(original_images)))
            print("Collaborative Image Inpainting SSIM : {}".format(s2/len(original_images)))
            original_method_images = (((asarray(original_method_images)+ 1) / 2.0)*255).astype(int)
            is_avg, is_std = self.calculate_inception_score(original_method_images)
            print("Semantic Image Inpainting IS : {} +- {}".format(is_avg, is_std))
            new_method_images = (((asarray(new_method_images)+ 1) / 2.0)*255).astype(int)
            is_avg, is_std = self.calculate_inception_score(new_method_images)
            print("Collaborative Image Inpainting IS : {} +- {}".format(is_avg, is_std))



    def find_latent_vectors(self, index_list=range(10), save_images = True):

        print("--------- START OFFLINE COMPUTING OF CLOSEST ENCODING VECTORS OF ALL TRAINING SET IMAGES ---------")

        if(save_images) : #Save all images in pickle files to reduce computation time for discriminator shaping
            all_imgs = []
            for i in tqdm(range(1000)) : #dis_shaping_batch * 1000 -> obtain 1000 batchs of #(dis_shaping batch) images each
                all_imgs.append(self.dataset.train_next_batch(self.flags.dis_shaping_batch))

            for i in tqdm(range(10)) : #save 10 files. For each file we have 100 batch of #(dis_shaping batch) images each -> #dis_shaping_batch * 1000 images in total
                t = all_imgs[i*100:(i+1)*100]
                with open("{}/images_{}.pkl".format(self.images_dir, i), 'wb') as f:
                    pickle.dump(t, f)

        #Once we have finished saving the 10 files. At each call, we will compute #dis_shaping_batch * 100 latent vector z,
        # given the corresponding indices in parameters

        for index in index_list : #might take long time to obtain. better choose 3 or 4 indices maximum at same time
            latent_vectors = []
            with open("{}/images_{}.pkl".format(self.images_dir, index), 'rb') as f:
                all_imgs = pickle.load(f)
                self.flags.sample_batch = self.flags.dis_shaping_batch #change batch for preprocessing model
                for i in tqdm(range(100)) :
                    imgs = all_imgs[i]  # take a batch of 60 imgs
                    z, _ = self.find_closest_encoding(self, for_discriminator_shaping=True, load=True, images=imgs)
                    latent_vectors.append(z)
                # Save the latent vector z in pkl file
                with open("{}/latent_vectors_{}.pkl".format(self.vectors_dir, index), 'wb') as f:
                    pickle.dump(latent_vectors, f)

        print("--------- FINISHED OFFLINE COMPUTING OF CLOSEST ENCODING VECTORS OF ALL TRAINING SET IMAGES ---------")
        
    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if(self.flags.earlier_stage > 0) :
                ckpt_name = 'model-{}'.format(self.flags.earlier_stage)
            else :
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(ckpt_name.split('-')[1])
            #self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0]) #A ENLEVER SI L AUTRE FONCTIONNE pour les 2

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')
            return True
        else:
            return False

