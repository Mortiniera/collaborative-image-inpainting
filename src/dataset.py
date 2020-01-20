import time
import numpy as np
import cv2

import utils as utils


class CelebA(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = [64, 64, 3]
        self.input_height = self.input_width = 108
        self.num_trains, self.num_vals = 0, 0

        self.celeba_train_path = "/{}/Data/{}/train".format(flags.root_folder, self.dataset_name)
        self.celeba_val_path = "/{}/Data/{}/val".format(flags.root_folder, self.dataset_name)
        self._load_celeba()

        np.random.seed(seed=int(time.time()))  # set random seed according to the current time

    def _load_celeba(self):
        print('Load {} dataset...'.format(self.dataset_name))

        self.train_data = utils.all_files_under(self.celeba_train_path)
        self.num_trains = len(self.train_data)

        self.val_data = utils.all_files_under(self.celeba_val_path)
        self.num_vals = len(self.val_data)
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs)

    def val_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.val_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs)

class fashion_mnist(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = [64, 64, 1]
        self.input_height = self.input_width = 28
        self.num_trains, self.num_vals = 0, 0

        self.fashion_train_path = "/{}/Data/{}/train".format(self.flags.root_folder, self.dataset_name)
        self.fashion_val_path = "/{}/Data/{}/val".format(self.flags.root_folder, self.dataset_name)
        self._load_fashion()

        np.random.seed(seed=int(time.time()))  # set random seed according to the current time

    def _load_fashion(self):
        print('Load {} dataset...'.format(self.dataset_name))
        
        self.train_data = utils.all_files_under(self.fashion_train_path)
        self.num_trains = len(self.train_data)

        self.val_data = utils.all_files_under(self.fashion_val_path)
        self.num_vals = len(self.val_data)
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width, is_gray_scale=True)
                      for batch_path in batch_paths]
        
        batch_imgs_ = [utils.random_flip(
            utils.transform(cv2.resize(batch_imgs[idx], (self.image_size[0], self.image_size[1]), 1)) )
            for idx in range(len(batch_imgs))]

        return np.asarray(batch_imgs)

    def val_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.val_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width, is_gray_scale=True)
                      for batch_path in batch_paths]

        batch_imgs_ = [utils.random_flip(
            utils.transform(cv2.resize(batch_imgs[idx], (self.image_size[0], self.image_size[1]), 1)))
            for idx in range(len(batch_imgs))]
            
        return np.asarray(batch_imgs)


# noinspection PyPep8Naming
def Dataset(flags, dataset_name):
    if dataset_name == 'celebA':
        return CelebA(flags, dataset_name)
    elif dataset_name == 'fashion_mnist':
        return fashion_mnist(flags, dataset_name)
    else:
        raise NotImplementedError

