import numpy as np
import copy

# width and height of images in dataset
WIDTH = 12
HEIGHT = 14

# number of distinct letters
NUM_OF_ALPHABET = 33 + 1  # UNICODE is the best invention the mankind has for russian letters *thumbs_up*

# seed for creating initial weights
np.random.seed(1337228)


# generation of initial weights and biases
def gen_params(NUM_OF_NEURONS):
    W1 = np.random.normal(size=(NUM_OF_NEURONS, WIDTH * HEIGHT)) * np.sqrt(1. / (WIDTH * HEIGHT))
    b1 = np.random.normal(size=(NUM_OF_NEURONS, 1)) * np.sqrt(1. / NUM_OF_NEURONS)
    W2 = np.random.normal(size=(NUM_OF_ALPHABET, NUM_OF_NEURONS)) * np.sqrt(1. / (NUM_OF_NEURONS * 2))
    b2 = np.random.normal(size=(NUM_OF_ALPHABET, 1)) * np.sqrt(1. / (WIDTH * HEIGHT))
    return W1, b1, W2, b2


# first activation function
def relu(z):
    return np.maximum(z, 0)


# derivative of relu
def relu_deriv(z):
    return z > 0


# second activation function
def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


# yep, that's a function, yep-yep
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


# getting all the predictions while teaching the NN
def get_predictions(a2):
    return np.argmax(a2, 0)


# calculating the accuracy of NN based on its predictions
def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


# main NN class
class Neural:
    def __init__(self, data, NUM_OF_NEURONS):
        # initial parameters
        self.W1, self.b1, self.W2, self.b2 = gen_params(NUM_OF_NEURONS)

        # dimensions of the dataset
        self.m, self.n = data.shape

        # reservation of first 2000 entries of dataset for accuracy checking
        self.data_dev = data[0:2000].T
        self.y_dev = self.data_dev[0]
        self.x_dev = self.data_dev[1:self.n]

        # taking the rest of the dataset for teaching purposes
        self.data_train = data[2000:self.m].T
        y_train = self.data_train[0]
        self.y_train = y_train.astype(int)
        self.x_train = self.data_train[1:self.n]

    # forward propagation method
    def forward_prop(self, x):
        z1 = self.W1.dot(x) + self.b1
        a1 = relu(z1)
        z2 = self.W2.dot(a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    # backward propagation method
    def backward_prop(self, z1, a1, z2, a2, x, y):
        one_hot_y = one_hot(y)
        dz2 = a2 - one_hot_y
        dW2 = 1 / self.m * dz2.dot(a1.T)
        db2 = 1 / self.m * np.sum(dz2)
        W2 = copy.deepcopy(self.W2)
        dz1 = W2.T.dot(dz2) * relu_deriv(z1)
        dW1 = 1 / self.m * dz1.dot(x.T)
        db1 = 1 / self.m * np.sum(dz1)
        return dW1, db1, dW2, db2

    # updating parameters after forward and backward propagations
    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2

    # gradient descent method
    def gradient_descent(self, x, y, alpha, iterations):
        for i in range(iterations):
            z1, a1, z2, a2 = self.forward_prop(x)
            dW1, db1, dW2, db2 = self.backward_prop(z1, a1, z2, a2, x, y)
            self.update_params(dW1, db1, dW2, db2, alpha)

            # checking accuracy every 10 iterations
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(a2)
                print(get_accuracy(predictions, y))
        return self.W1, self.b1, self.W2, self.b2

    # getting predictions of NN
    def make_predictions(self, X):
        _, _, _, a2 = self.forward_prop(X)
        predictions = get_predictions(a2)
        return predictions
