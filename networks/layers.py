class Layer:
    def __init__(self, size):
        self.size = size



class InputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)
        self.a = []



class OutputLayer(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.activation = activation
        self.z = []
        self.a = []
        self.b = []
        self.da = []
        self.dl = 0
        self.weights = []


class HiddenLayer(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.activation = activation
        self.z = []
        self.a = []
        self.b = []
        self.da = []
        self.weights = []

