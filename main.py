import numpy as np
from matplotlib import pyplot as plt
import threading
import time

from neural import Neural

# offset for conversion of characters into UNICODE and vice versa
UNICODE_OFFSET = 1072


# numpy dataset load
def prepare_data():
    matrix = np.load('data.npy')
    return matrix


# main NN function
def create_neural(data, neurons, it, accuracy):
    # creation of NN
    neural = Neural(data, neurons)

    # gradient descent method to teach the NN
    neural.W1, neural.b1, neural.W2, neural.b2 = neural.gradient_descent(neural.x_train, neural.y_train, 0.10, it)

    # accuracy check of the trained NN
    cnt = 0
    for i in range(2000):
        prediction = neural.make_predictions(neural.x_dev[:, i, None])
        if prediction != neural.y_dev[i]:
            cnt += 1
    accuracy.append((1 - cnt / 2000) * 100)


# NN save
def save_neural(neural):
    np.save('W1', neural.W1)
    np.save('W2', neural.W2)
    np.save('b1', neural.b1)
    np.save('b2', neural.b2)


# NN load
def load_neural(neural):
    neural.W1 = np.load('W1.npy')
    neural.W2 = np.load('W2.npy')
    neural.b1 = np.load('b1.npy')
    neural.b2 = np.load('b2.npy')


# main NN function
def main():
    # initial time stamp for convenience
    start_time = time.time()

    # NN creation, teaching and accuracy check
    data = prepare_data()
    threads = list()

    # list of amount of neurons of NN to check accuracy on various number of iterations
    neurons = [5, 10, 25, 35, 50]

    # main loop for NN
    for neuron in neurons:
        accuracy = list()
        iterations = list()
        for i in range(25, 301, 25):
            # creation of multiple threads for each amount of iterations
            x = threading.Thread(target=create_neural, args=(data, neuron, i, accuracy))
            threads.append(x)
            x.start()
            iterations.append(i)
        for j in range(len(threads)):
            # waiting for all the threads to finish their iteration
            threads[j].join()
        # plotting the results of NN accuracy check with specific amount of neurons
        plt.plot(iterations, accuracy, label=f"{neuron} neurons")

    # time stamp after all the NN's accuracies are checked
    elapsed_time = round(time.time() - start_time, 2)

    # time taken to do the work
    print("Time elapsed: " + str(int(elapsed_time // 60)) + " minutes and "
          + str(int(elapsed_time - 60 * (elapsed_time // 60))) + " seconds")

    # show plot
    plt.legend()
    plt.show()
    return None


# main check
if __name__ == '__main__':
    main()
