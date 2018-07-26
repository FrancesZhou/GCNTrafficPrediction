import numpy as np

class MinMaxNormalization01(object):
	def __init__(self, ):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("min: ", self._min, "max:", self._max)

	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)
		#return real_loss

class MinMaxNormalization01_minus_mean(object):
	def __init__(self, period=24*7):
		self.period = period
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("min: ", self._min, "max:", self._max)
		#
		d_shape = np.array(data).shape
		self._mean = np.zeros((self.period, d_shape[1], d_shape[2]))
		index = np.arange(d_shape[0]/self.period)
		for i in xrange(self.period):
			self._mean[i] = np.mean(data[self.period*index+i], axis=0)
		self._mean = 1. * (self._mean - self._min) / (self._max - self._min)

	def transform(self, data, pre_index=0):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		mean_index = (np.arange(len(norm_data)) + pre_index)%self.period
		norm_minus_mean_data = norm_data - self._mean[mean_index]
		return norm_data, norm_minus_mean_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data, mean_index=0, if_add_mean=True):
		if if_add_mean:
			index = mean_index%self.period
			inverse_norm_data = 1. * (data+self._mean[index]) * (self._max - self._min) + self._min
		else:
			inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

class MinMaxNormalization01_by_axis(object):
	def __init__(self):
		pass
	def fit(self, data):
		self._min = np.amin(data, axis=0)
		self._max = np.amax(data, axis=0)
	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		return norm_data
	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)
	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data
	#def real_loss(self, loss):

class MinMaxNormalization_neg_1_pos_1(object):
	def __init__(self):
		pass

	def fit(self, X):
		self._min = X.min()
		self._max = X.max()
		print("min:", self._min, "max:", self._max)

	def transform(self, X):
		X = 1. * (X - self._min) / (self._max - self._min)
		X = X * 2. - 1.
		return X

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

	def inverse_transform(self, X):
		X = (X + 1.)/2.
		X = 1. * X * (self._max - self._min) + self._min
		return X

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)/2.
	#return real_loss
