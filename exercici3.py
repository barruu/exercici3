import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Configuració de la xarxa neuronal
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 2

        # Inicialització dels pesos i sesgos amb valors aleatoris
        self.input_layer_weights = np.random.rand(self.input_size, self.hidden_size)
        self.input_layer_bias = np.random.rand(1, self.hidden_size)

        self.output_layer_weights = np.random.rand(self.hidden_size, self.output_size)
        self.output_layer_bias = np.random.rand(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Capa oculta
        self.hidden_layer_input = np.dot(X, self.input_layer_weights) + self.input_layer_bias
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Capa de sortida
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_layer_weights) + self.output_layer_bias
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            # Feedforward
            hidden_layer_input = np.dot(X, self.input_layer_weights) + self.input_layer_bias
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.output_layer_weights) + self.output_layer_bias
            output_layer_output = self.sigmoid(output_layer_input)

            # Càlcul d'errors
            error = y - output_layer_output

            # Backpropagation
            d_output = error * self.sigmoid_derivative(output_layer_output)
            error_hidden_layer = d_output.dot(self.output_layer_weights.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            # Actualització de pesos i sesgos
            self.output_layer_weights += hidden_layer_output.T.dot(d_output) * learning_rate
            self.output_layer_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.input_layer_weights += X.T.dot(d_hidden_layer) * learning_rate
            self.input_layer_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Crear una instància de la xarxa neuronal
nn = NeuralNetwork()

# Dades d'entrenament per a XOR i AND
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])

# Entrenar la xarxa neuronal amb backpropagation
nn.train(X_train, y_train, learning_rate=0.1, epochs=10000)

# Realitzar l'alimentació cap endavant per a les entrades d'entrenament
output = nn.feedforward(X_train)

print("Sortida desitjada per a les entrades [0, 0], [0, 1], [1, 0], [1, 1]:", output)