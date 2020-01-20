import os
import tensorflow as tf

from inpaint_solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('dataset', 'fashion_mnist', 'dataset to choose [celebA|fashion_mnist], default: fashion_mnist')
tf.flags.DEFINE_string('root_folder', 'content', 'root folder of directory hierarchy, default: content for Google Colab')


tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate to update latent vector z, default: 0.01')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum term of the NAG optimizer for latent vector, default: 0.9')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')
tf.flags.DEFINE_float('lamb', 0.0001, 'hyper-parameter for prior loss, default: 0.0001')
tf.flags.DEFINE_string('mask_type', 'center', 'mask type choice in [center|random|half|pattern], default: center')
tf.flags.DEFINE_integer('img_size', 64, 'image height or width, default: 64')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size for one feed forward, default: 64')


tf.flags.DEFINE_float('learning_rate_adam', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer("rollout_steps", 50, 'rollout steps for sample refinement", default : 50')
tf.flags.DEFINE_float("rollout_rate", 0.1, 'rollout rate for sample refinement, default : 0.1')
tf.flags.DEFINE_integer("refiner_batch", 4, 'sample batch size to use for testing refiner", default : 4')
tf.flags.DEFINE_integer("dis_shaping_batch", 60, 'sample batch size to use for discriminator shaping : Chosen as (dataset_training_size / 1000)", default : 60')


tf.flags.DEFINE_string('mode', 'inpaint', 'choice of mode to use between [standard | inpaint], default: inpaint')
tf.flags.DEFINE_bool('observe_evolution', True, 'save refined samples at different refinement steps, default: true')


tf.flags.DEFINE_bool('heatmap', True, 'generate a heatmap to highlight differences between previous and current method, default: true')

tf.flags.DEFINE_integer('iters', 1000, 'number of iterations to optimize latent vector, default: 1500')
tf.flags.DEFINE_integer('num_try', 5, 'number of random samples, default: 5')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('sample_batch', 4, 'number of sampling images, default: 2')
tf.flags.DEFINE_string('load_model', None,
                       'saved DCGAN model that you wish to test, (e.g. 20180704-1736), default: None')

tf.flags.DEFINE_integer('earlier_stage', 0, 'load model at earlier saved checkpoint iteration, (e.g. 4499), default: 0')

tf.flags.DEFINE_integer('test_number', 0, 'indice of pair of previously saved batch images and vectors to test, default: 0')
tf.flags.DEFINE_bool('print_ds_loss', False, 'Print loss during discriminator shaping, default: False')
tf.flags.DEFINE_bool('overwrite', False, 'overwrite previously saved vector by the new one, default: false')
tf.flags.DEFINE_bool('load_test_vector', True, 'load an already saved latent vector, default: true')
tf.flags.DEFINE_bool('offline', False, 'offline computing of closest encoding vectors of training set , default: false')




def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)

    if(FLAGS.offline) :
        solver.find_latent_vectors()
    else :
        solver.collaborative_sampling_inpainting()


if __name__ == '__main__':
    tf.app.run()
