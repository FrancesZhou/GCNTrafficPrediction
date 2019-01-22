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