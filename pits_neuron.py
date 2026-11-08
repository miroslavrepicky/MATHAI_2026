# TODO make a forward function in network of pits neurons when you can see separate neurons and weights.
# NOTE! cant use neurons and connections because we cannot guarantee order of forwarding.
# NOTE! using numpy and layers we are loosing information of order of neurons and specific neurons
# TODO set specific weights to specific connections between specific neurons while keeping order of forward on layers
from pywin.framework.scriptutils import lastArgs


class Layer:
    def __init__(self, size):
        self.size = size
        self.weights = []
        self.activations = []
        self.zs = []

class InputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)

class OutputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)


class PitsModel:
    def __init__(self):
        self.layers = []
        self.activation_function = ActivationFuction()

    def add_layer(self, layer):
        if len(self.layers) == 0:
            if type(layer) is not InputLayer:
                raise TypeError('First layer must be an InputLayer object')

        self.layers.append(layer)

    def reset_model(self):
        for layer in self.layers:
            layer.zs = []
            layer.activations = []

    def forward(self, inputs:list):
        if len(inputs) != self.layers[0].size:
            raise TypeError('Wrong size of input')
        activation = inputs

        for i, layer in enumerate(self.layers):
            print(activation)
            if isinstance(layer, InputLayer):
                continue
            # [1, 1]
            #       [[1, 1],
            #       [-2, -1]]
            z = 0
            for k in range(layer.size):
                z = 0
                vypocet = ""
                for l in range(len(layer.weights[k])):
                    vypocet += str(layer.weights[k][l]) + ' * ' + str(activation[l])
                    vypocet += " + "
                    z += activation[l]*layer.weights[k][l]
                print(vypocet[:-2])
                layer.zs.append(z)
                layer.activations.append(self.activation_function.forward(z))

            activation = layer.activations

    def set_weights(self, weights):
        weight_counter = 0
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Determine dimensions
            if isinstance(current_layer, InputLayer):
                input_dim = current_layer.size
            else:
                input_dim = current_layer.size

            if isinstance(next_layer, OutputLayer):
                output_dim = next_layer.size
            else:
                output_dim = next_layer.size

            # input dim = pocet stlpcov
            # output dim = pocet riadkov
            for j in range(output_dim):
                next_layer.weights.append([])
                for k in range(input_dim):
                    next_layer.weights[j].append(weights[weight_counter])
                    weight_counter += 1


            #next_layer.weights = np.random.randn(input_dim, output_dim) * 0.1

    def print_weights(self):
        for layer in self.layers:
            print(layer.weights)

    def print_results(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}: zs = {layer.zs}; activations = {layer.activations}')



class ActivationFuction:
    def forward(self, z):
        if z >= 2:
            return 1
        else:
            return 0




if __name__ == '__main__':
    model = PitsModel()
    model.add_layer(InputLayer(2))
    model.add_layer(Layer(2))
    model.add_layer(Layer(1))
    model.set_weights([1, 1, -2, -1, 2, 2])
    model.forward([1, 1])
    model.print_results()

