class Model:
    def __init__(self, size):
        self.size = size
        self.weights = [0 for _ in range(size)]
        self.bias = 0
        self.activation_func = ActivationFunction()
        self.theta = 0.2

    def forward(self, X):
        print("===========FORWARD===========")
        for x in X:
            z = 0
            for i in range(self.size):
                z += (self.weights[i] * x[i])
            z += self.bias
            activation = self.activation_func.forward(z, self.theta)
            print(f'For input: {x}; z: {z}; output: {activation}. ')

    def train_forward(self, x):
        z = 0
        for i in range(self.size):
            z += (self.weights[i] * x[i])
        z += self.bias
        activation = self.activation_func.forward(z, self.theta)
        return activation

    def fit(self, X, y, alpha=1):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X[0]) != self.size:
            raise ValueError("Number of one set of inputs must be equal to size")
        training = True
        epoch = 1
        while training:
            training = False
            print(f"Epoch: {epoch}")
            epoch += 1
            for i in range(len(X)):
                activation = self.train_forward(X[i])
                if activation != y[i]:
                    training = True
                    for k in range(self.size):
                        self.weights[k] += alpha*X[i][k]*y[i]
                    self.bias += alpha*y[i]

                print(f'For input {X[i]}; out: {activation} and target {y[i]}. New weights and bias: {self.weights, self.bias} ')

class ActivationFunction:
    def forward(self, z, theta):
        if z > theta:
            return 1
        elif z < -theta:
            return -1
        else:
            return 0


if __name__ == "__main__":
    model = Model(2)
    X = [[1, 1], [1, 0], [0, 1], [0, 0]]
    y = [1, -1, -1, -1]
    model.fit(X, y)
    model.forward(X)