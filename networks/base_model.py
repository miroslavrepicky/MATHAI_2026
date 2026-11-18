from layers import *
from activations import *
from loss_funtions import *
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from os.path import join


class BaseModel:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def initialize_weights(self, mode="random"):
        if mode == "zeros":
            raise NotImplementedError
        elif mode == "zadanie":
            # sieť 4 -> 2 (sigmoid) -> 1 (lineárny); hodnoty zo zadania
            hidden, out = self.layers[1], self.layers[2]
            # riadok 1 = váhy x1..x4 -> h1, riadok 2 = x1..x4 -> h2; weights má tvar (4, 2)
            hidden.weights = np.array([(0.20, -0.20, -0.40, -0.10),
                                       (-0.30, -0.60, 0.70, 0.40)]).T
            hidden.b = np.array([-0.14, 0.03])  # b1, b2
            out.weights = np.array([[0.20], [0.60]])  # v1, v2 -> tvar (2, 1)
            out.b = np.array([0.07])  # c
            for layer in (hidden, out):
                layer.a = np.zeros(layer.size)
                layer.da = np.zeros(layer.size)
                layer.z = np.zeros(layer.size)
        elif mode == "ones":
            raise NotImplementedError
        elif mode == "random":
            np.random.seed(42)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, InputLayer):
                    continue
                layer.weights = np.random.uniform(-0.5, 0.5, (self.layers[i - 1].size, layer.size))
                # layer.b = np.random.uniform(-0.5, 0.5, layer.size)
                layer.b = np.zeros(layer.size)
                layer.a = np.zeros(layer.size)
                layer.da = np.zeros(layer.size)
                layer.z = np.zeros(layer.size)
        else:
            raise NotImplementedError("Not implemented weight initialization mode")

    def build(self, mode="random"):
        self.initialize_weights(mode=mode)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer):
                layer.a = x
                continue
            prev_layer = self.layers[i - 1]
            layer.z = np.dot(prev_layer.a, layer.weights) + layer.b
            layer.a = layer.activation.forward(layer.z)
        return self.layers[-1].a

    def backward(self, x, y, alpha):
        last = len(self.layers) - 1
        self.layers[last].da = self.loss.backward(self.layers[last].a, y)

        # 1. delty pre všetky vrstvy (so starými váhami)
        for i in range(last, 0, -1):
            layer = self.layers[i]
            if i == last:
                upstream = layer.da
            else:
                nxt = self.layers[i + 1]
                upstream = np.dot(nxt.da, nxt.weights.T)
            d = layer.activation.backward(layer.z)
            layer.da = d @ upstream if d.ndim == 2 else upstream * d

        # 2. aktualizácia váh a biasov
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev = self.layers[i - 1]
            layer.weights -= alpha * np.outer(prev.a, layer.da)
            layer.b -= alpha * layer.da


    def fit(self, X, Y, epochs=10000, learning_rate=0.1, loss=CCE()):
        self.loss = loss
        for epoch in range(epochs):
            # 1. FORWARD
            # 2. LOSS
            # 3. BACKPROP
            for x, y in zip(X, Y):
                output = self.forward(x)
                # error = [loss.forward(output[o], y[o]) for o in range(len(output))]
                error = loss.forward(output, y)
                print("Epoch:", epoch)
                print("Error:", error)
                self.backward(x, y, learning_rate)
        for x, y in zip(X, Y):
            output = self.forward(x)
            retazec = f"X: {x}, "
            for i, o in enumerate(output):
                retazec += f"Y{i+1}: {o:.6f} | "
            print(retazec)






if __name__ == "__main__":
    model = BaseModel()
    model.add_layer(InputLayer(784))
    model.add_layer(HiddenLayer(128, Sigmoid()))
    model.add_layer(OutputLayer(10, Softmax()))
    model.build()
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Y = np.array([0, 1, 1, 0])
    # model.fit(np.array([(1, 1), (1, 0), (0, 1), (0, 0)]), np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
