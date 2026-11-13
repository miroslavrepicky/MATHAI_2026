from traceback import print_tb

import matplotlib.pyplot as plt

class Model:
    def __init__(self, size):
        self.size = size
        self.weights = [0 for _ in range(size)]
        self.bias = 0
        self.activation_func = ActivationFunction()
        self.theta = 0.0
        self.tolerance = 0.001

    def forward(self, X):
        print("===========FORWARD===========")
        for x in X:
            z = 0
            for i in range(self.size):
                z += (self.weights[i] * x[i])
            z += self.bias
            activation = self.activation_func.forward(z)
            print(f'For input: {x}; z: {z}; output: {activation}. ')

    def train_forward(self, x):
        z = 0
        for i in range(self.size):
            z += (self.weights[i] * x[i])
        z += self.bias
        return z

    def fit(self, X, y, alpha=0.2):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X[0]) != self.size:
            raise ValueError("Number of one set of inputs must be equal to size")
        training = True
        epoch = 1
        errors = []
        while training and epoch < 200:
            training = False
            print(50 * '=')
            print(f"Epoch: {epoch}")
            epoch += 1
            error = 0
            for i in range(len(X)):
                deltas = []
                activation = self.train_forward(X[i])
                for k in range(self.size):
                    delta_w = alpha * X[i][k] * (y[i] - activation)
                    deltas.append(delta_w)
                    self.weights[k] += delta_w
                delta_b = alpha * (y[i] - activation)
                deltas.append(delta_b)
                self.bias += delta_b
                print(f'INPUT: {X[i]} | NET: {activation} | TARGET: {y[i]} | ZMENY VAH: {deltas} | NOVE VAHY: {self.weights + [self.bias]}')
                error += ((self.train_forward(X[i]) - y[i]) ** 2)
            if error > self.tolerance:
                training = True
            print(error)
            errors.append(error)

        # epps = [i for i in range(len(errors))]
        # plt.plot(epps[15:], errors[15:])
        # plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        # plt.show()


class ActivationFunction:
    def forward(self, z):
        if z >=0:
            return 1
        else:
            return -1


if __name__ == "__main__":
    model = Model(2)
    X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y = [1, -1, -1, -1]
    model.fit(X, y)
    model.forward(X)