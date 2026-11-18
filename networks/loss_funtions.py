import numpy as np


class Loss:
    def forward(self, activation, y):
        raise NotImplementedError

    def backward(self, activation, y):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, activation, y):
        return 0.5*(activation - y)**2

    def backward(self, activation, y):
        return activation - y

class CCE(Loss):
    def forward(self, activation, y):
        return -np.sum(y * np.log(np.clip(activation, 1e-12, 1)))

    def backward(self, activation, y):
        return -y / np.clip(activation, 1e-12, 1)   # dL/dŷ