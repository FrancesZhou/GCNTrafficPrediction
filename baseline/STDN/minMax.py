import numpy as np

class MinMax:
    def __init__(self, data):
        self.max_value = np.max(data)
        self.min_value = np.min(data)
        self.data = data

    def transform(self):
        return (self.data-self.min_value)/(self.max_value-self.min_value)

    def inverse(self, rmse):
        return rmse * (self.max_value - self.min_value)

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

    def inverse(self, data):
        inverse_norm_data = 1. * data * (self._max - self._min) + self._min
        return inverse_norm_data

    def real_loss(self, loss):
        # loss is rmse
        return loss*(self._max - self._min)
        #return real_loss