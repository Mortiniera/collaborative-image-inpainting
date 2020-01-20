"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""

from __future__ import print_function
import os
import argparse
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['fashion_mnist'],
                    help='name of dataset to download [fashion_mnist]')


def prepare_data_dir(path='./Data'):
    if not os.path.exists(path):
        os.mkdir(path)

def download_fashion_mnist(dirpath) :
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    folder = dirpath + "/fashion_mnist/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    train_folder = folder + "train"
    val_folder = folder + "val"
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    # TRAIN_IMAGES
    for i in tqdm(range(len(train_images))):
        im = Image.fromarray((train_images[i]).astype(np.uint8))
        im.save(train_folder + "/train_" + str(i) + ".png")

    # TEST_IMAGES
    for i in tqdm(range(len(test_images))):
        im = Image.fromarray((test_images[i]).astype(np.uint8))
        im.save(val_folder + "/val_" + str(i) + ".png")

    print("SUCESSFULLY DOWNLOADED FASHION MNIST DATA SET ! ")

if __name__ == '__main__':
    args = parser.parse_args()
    prepare_data_dir()

    if 'fashion_mnist' in args.datasets:
        print("FASHION_MNIST")
        download_fashion_mnist('./Data')   # download fashion mnist dataset