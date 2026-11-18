import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


# NOTE: A' = A * (1 - A)

class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))


class ReLU(Activation):
    def forward(self, x):
        return np.where(x > 0, x, 0)


    def backward(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(Activation):
    def forward(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def backward(self, x):
        s = self.forward(x)
        return np.diag(s) - np.outer(s, s)   # Jacobián: s_i * (δ_ij - s_j)

class Linear(Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)