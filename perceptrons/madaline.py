from prettytable import PrettyTable

class Model:
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.weights = []
        self.biases = []
        self.output_weights = [0.5 for _ in range(size)]
        self.output_bias = 0.5
        self.activation_func = ActivationFunction()
        self.initialize_weights()
        self.zzz = [[f'Z{i+1}_in', f'Z{i+1}'] for i in range(self.size)]
        self.things_to_print = [f'X_{i}' for i in range(input_size)] + [znak for retazec in self.zzz for znak in retazec] + ["Y_in"] + ["Y"] + ["Vypocet a zmeny vah"] + ["Nove vahy a bias"]
        self.t = PrettyTable(self.things_to_print)


    def initialize_weights(self, w=0, b=0):
        for i in range(self.size):
            self.weights.append([w for _ in range(self.input_size)])
            self.biases.append(b)


    def fit(self, X, targets, alpha=0.2):
        if len(X[0]) != self.input_size:
            raise ValueError("Input has wrong size")
        training = True
        epoch = 0
        while training and epoch < 2:
            self.zzz = [[f'Z{i + 1}_in', f'Z{i + 1}'] for i in range(self.size)]
            self.things_to_print = [f'X_{i}' for i in range(self.input_size)] + [znak for retazec in self.zzz for znak in
                                                                            retazec] + ["Y_in"] + ["Y"] + [
                                       "Vypocet a zmeny vah"] + ["Nove vahy a bias"]
            self.t = PrettyTable(self.things_to_print)
            epoch += 1
            print(f'Epoch: {epoch}')
            for c, x in enumerate(X):
                z_ins = []
                zs = []
                for i in range(self.size):
                    z = 0
                    for k in range(self.input_size):
                        z += self.weights[i][k] * x[k]
                    z += self.biases[i]
                    z_ins.append(z)
                    zs.append(self.activation_func.forward(z))
                y_in = 0
                for i in range(self.size):
                    y_in += self.output_weights[i] * zs[i]
                y_in += self.output_bias
                y = self.activation_func.forward(y_in)
                if y != targets[c]:
                    if targets[c] == 1:
                        absolute = [abs(v) for v in z_ins]
                        lowest_value = min(absolute)
                        z_index = absolute.index(lowest_value)
                        for e in range(self.input_size):
                            delta_w = alpha * (1 - z_ins[z_index]) * x[e]
                            self.weights[z_index][e] += delta_w
                        delta_b = alpha * (1 - z_ins[z_index])
                        self.biases[z_index] += delta_b
                        zss = []
                        for i in range(len(z_ins)):
                            zss.append(z_ins[i])
                            zss.append(zs[i])
                        zoznam = [] + x + zss + [y_in] + [y] + [f'y != t && t = -1; Vahy sme menili pre: Z_{z_index+1}'] + [str(self.weights) + " " + str(self.biases)]
                        self.t.add_row(zoznam)
                    elif targets[c] == -1:
                        vrstvy = "y != t && t = -1; Vahy sme menili pre: "
                        for z_i, z_net in enumerate(z_ins):
                            if z_net >= 0:
                                vrstvy += f'Z_{z_i+1}; '
                                for e in range(self.input_size):
                                    delta_w = alpha * (-1 - z_net) * x[e]
                                    self.weights[z_i][e] += delta_w
                                delta_b = alpha * (-1 - z_net)
                                self.biases[z_i] += delta_b
                        zss = []
                        for i in range(len(z_ins)):
                            zss.append(z_ins[i])
                            zss.append(zs[i])
                        zoznam = [] + x + zss + [y_in] + [y] + [vrstvy] + [str(self.weights) + " " + str(self.biases)]
                        self.t.add_row(zoznam)
                    else:
                        raise ValueError("You trippin hitta or wha?! Targets can only be -1 or 1!")
                else:
                    zss = []
                    for i in range(len(z_ins)):
                        zss.append(z_ins[i])
                        zss.append(zs[i])
                    zoznam = [] + x + zss + [y_in] + [y] + ["Bez aktualizacie vah"] + [str(self.weights) + " " + str(self.biases)]
                    self.t.add_row(zoznam)
            self.t.add_divider()
            print(self.t)



class ActivationFunction:
    def forward(self, z):
        if z >= 0:
            return 1
        else:
            return -1


if __name__ == "__main__":
    model = Model(2, 2)
    X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y = [1, -1, -1, -1]
    model.fit(X, y)


