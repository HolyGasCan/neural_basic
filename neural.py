import numpy
import numpy as np
import copy

WIDTH = 12
HEIGHT = 14

NUM_OF_NEURONS = 20
NUM_OF_ALPHABET = 33 + 1   # UNICODE - лучшее изобретение человечества

B = 0.50


def gen_params():
    W1 = np.random.normal(size=(NUM_OF_NEURONS, WIDTH * HEIGHT)) * np.sqrt(1. / (WIDTH * HEIGHT))
    b1 = np.random.normal(size=(NUM_OF_NEURONS, 1)) * np.sqrt(1. / NUM_OF_NEURONS)
    W2 = np.random.normal(size=(NUM_OF_ALPHABET, NUM_OF_NEURONS)) * np.sqrt(1. / (NUM_OF_NEURONS * 2))
    b2 = np.random.normal(size=(NUM_OF_ALPHABET, 1)) * np.sqrt(1. / (WIDTH * HEIGHT))
    # W1 = np.random.rand(NUM_OF_NEURONS, WIDTH * HEIGHT) - B
    # W2 = np.random.rand(NUM_OF_ALPHABET, NUM_OF_NEURONS) - B
    # b1 = np.random.rand(NUM_OF_NEURONS, 1) - B
    # b2 = np.random.rand(NUM_OF_ALPHABET, 1) - B
    np.save('W1_init', W1)
    np.save('b1_init', b1)
    np.save('W2_init', W2)
    np.save('b2_init', b2)


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


def relu(z):
    return np.maximum(z, 0)


def relu_deriv(z):
    return z > 0


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


class Neural:
    def __init__(self, data):
        # gen_params()
        # np.random.shuffle(data)
        # np.save('data', data)

        self.m, self.n = data.shape

        self.W1 = np.load('W1_init.npy')
        self.W2 = np.load('W2_init.npy')
        self.b1 = np.load('b1_init.npy')
        self.b2 = np.load('b2_init.npy')

        self.data_dev = data[0:2000].T
        self.y_dev = self.data_dev[0]
        self.x_dev = self.data_dev[1:self.n]

        self.data_train = data[2000:self.m].T
        y_train = self.data_train[0]
        self.y_train = y_train.astype(int)
        self.x_train = self.data_train[1:self.n]

    def forward_prop(self, x):
        z1 = self.W1.dot(x) + self.b1
        a1 = relu(z1)
        z2 = self.W2.dot(a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

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

    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2

    def gradient_descent(self, x, y, alpha, iterations):
        for i in range(iterations):
            z1, a1, z2, a2 = self.forward_prop(x)
            dW1, db1, dW2, db2 = self.backward_prop(z1, a1, z2, a2, x, y)
            self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(a2)
                print(get_accuracy(predictions, y))
        return self.W1, self.b1, self.W2, self.b2

    def make_predictions(self, X):
        _, _, _, a2 = self.forward_prop(X)
        predictions = get_predictions(a2)
        return predictions

    '''
    def test_prediction(self, index):
        current_image = self.x_train[:, index, None]
        prediction = self.make_predictions(self.x_train[:, index, None])
        label = self.y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
    '''
