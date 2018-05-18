import numpy as np

class MultilayerPerceptron():
    
    # Starting object setting the neuron weights
    def __init__(self, input, hidden, output):
        if len(hidden) == 0:
             raise TypeError('Neural Net MLP invalid!')

        np.random.seed(1)
        self.weights = [2 * np.random.random((input, hidden[0])) - 1]
        if len(hidden) > 1:
            for i in range(1, len(hidden)):
                self.weights.append(2 * np.random.random((hidden[i - 1], hidden[i])) - 1)
            self.weights.append(2 * np.random.random((hidden[len(hidden) - 1], output)) - 1)
        else:
            self.weights.append(2 * np.random.random((hidden[0], output)) - 1)

    # Method to adjust neurons weight
    def __sigmoid_derivative(self, inputs):
        return inputs * (1 - inputs)

    # Activation method
    def __sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    # Process to calculate the final result each neuron
    def __forward(self, inputs):
        w = [inputs, self.__sigmoid(np.dot(inputs, self.weights[0]))]
        for i in range(1, len(self.weights)):
            w.append(self.__sigmoid(np.dot(w[len(w) - 1], self.weights[i])))
        return w

    # This method will train the neurons to predict new values
    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            # forward-propagation
            w = self.__forward(inputs)

            # Checking error
            error = outputs - w[len(w) - 1]
            if (iteration % 1000) == 0:
                print ("Error: {}".format(np.mean(np.abs(error))))

            # back-propagation
            w_delta = [error * self.__sigmoid_derivative(w[len(w) - 1])]
            for i in range(len(w) - 1, -1, -1):
                if i > 1:
                    w_error = w_delta[len(w_delta) - 1].dot(self.weights[i - 1].T)
                    w_delta.append(w_error * self.__sigmoid_derivative(w[i - 1]))

            # Updating weights
            self.weights[0] += np.dot(inputs.T, w_delta[len(w_delta) - 1])
            j = len(w_delta) - 2
            for i in range(1, len(self.weights)):
                self.weights[i] += np.dot(w[i].T, w_delta[j])
                j -= 1

    # This method will execute the neural net
    def run(self, inputs):
        w = self.__forward(inputs)
        return w[len(w) - 1]

if __name__ == "__main__":
    nn = MultilayerPerceptron(2, [4, 3, 3], 1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    nn.train(X, y, 40000)
    print ("Final result: {}".format(nn.run(np.array([1, 0]))))