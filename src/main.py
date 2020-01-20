import os
import tensorflow as tf
import numpy as np
import pickle

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size for one feed forwrad, default: 64')
tf.flags.DEFINE_string('dataset', 'fashion_mnist', 'dataset to choose [celebA|fashion_mnist], default: fashion_mnist')

tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: False')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')

tf.flags.DEFINE_integer("rollout_steps", 50, 'rollout steps for sample refinement", default : 50')
tf.flags.DEFINE_float("rollout_rate", 0.1, 'rollout rate for sample refinement, default : 0.1')
tf.flags.DEFINE_integer("refiner_batch", 4, 'sample batch size to use for testing refiner", default : 4')
tf.flags.DEFINE_integer("dis_shaping_batch", 60, 'sample batch size to use for discriminator shaping : Chosen as (dataset_training_size / 1000)", default : 60')
tf.flags.DEFINE_string('mode', 'inpaint', 'choice of mode to use between [standard | inpaint], default: inpaint')
tf.flags.DEFINE_bool('observe_evolution', True, 'save refined samples at different refinement steps, default: true')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 64, 'number of sampling images from generator distribution, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    sample_z = None
    if(FLAGS.load_model) :
        with open("/content/collaborative-image-inpainting/fixed_z.pkl", 'rb') as f:
            sample_z = pickle.load(f)
    else :
        fixed_z = np.random.uniform(-1., 1., size=[FLAGS.sample_batch, FLAGS.z_dim])
        with open("/content/collaborative-image-inpainting/fixed_z.pkl", 'wb') as f:
            pickle.dump(sample_z, f)
    
    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train(fixed=True, fixed_z=sample_z)
    else:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
