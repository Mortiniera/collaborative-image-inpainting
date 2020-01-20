from __future__ import division
import numpy as np
import tensorflow as tf

class PolicyAdaptive(object):
	"""docstring for PolicyAdaptive"""
	def __init__(self, step_size, method):
		self.method = method
		self.lambda_ = step_size
		self.alpha_ = 0.9
		self.momentum = None
		# higher order 
		self.beta1_ = 0.9
		self.beta2_ = 0.5
		self.beta3_ = 0.5
		self.degree_ = 2
		self.eps_ = 1e-8
		self.mean_square = None
		self.loss = None

	def reset_moving_average(self):
		self.momentum = None
		self.mean_square = None
		self.loss = None

	def apply_gradient(self, theta, grad, loss=None):
		if self.method == 'sgd':
			theta -= self.lambda_ * grad
			return theta
		elif self.method == 'momentum':
			if self.momentum is not None:
				self.momentum = self.alpha_ * self.momentum + self.lambda_ * grad
			else:
				self.momentum = self.lambda_ * grad
			theta -= self.momentum
			return theta
		elif self.method == 'ladam':
			if self.momentum is not None:
				self.momentum = self.beta1_ * self.momentum + (1. - self.beta1_) * grad
			else:
				self.momentum = grad
			if self.mean_square is not None:
				self.mean_square = self.beta2_ * self.mean_square + (1. - self.beta2_) * grad**2
			else:
				self.mean_square = grad**2
			if self.loss is not None:
				self.loss = self.beta3_ * self.loss + (1. - self.beta3_) * loss
			else:
				self.loss = loss

			if theta.shape[1] > 2:	
				batch_shape = theta.shape
				dX = tf.reshape(self.lambda_ * self.momentum / (tf.sqrt(self.mean_square) + self.eps_), (batch_shape[0],-1))
				rescale = tf.reshape(tf.clip_by_value((self.loss + 0.5), clip_value_min=0.0, clip_value_max=10000.0), (batch_shape[0],1))
				theta = tf.reshape(theta, (batch_shape[0],-1))
				theta -= dX * (rescale ** self.degree_)
				theta = tf.reshape(theta, batch_shape)
			else:
				theta -= self.lambda_ * self.momentum / (np.sqrt(self.mean_square) + self.eps_) * (np.expand_dims( (self.loss+0.5).clip(min=0.0), axis=1) ** self.degree_)
			return theta			
		else:
			raise NotImplementedError
