import numpy as np
from matplotlib import pyplot as plt
import threading
import time

from prepare_data import PreparationData
from neural import Neural, get_accuracy

UNICODE_OFFSET = 1072

def preparation_data():
    # data_path = './data'
    # prepare_data = PreparationData(data_path)
    # prepare_data.rename_files('pic')
    # prepare_data.create_dataset()
    # matrix = prepare_data.create_matrix()
    matrix = np.load('data.npy')
    # print(np.linalg.matrix_rank(matrix))
    # draw(matrix, 30)
    return matrix


def draw(matrix, index):
    current_image = matrix[index]
    # label = chr(int(current_image[0]))
    current_image = current_image[1:169]
    current_image = current_image.reshape((14, 12)) * 255
    plt.imshow(current_image, interpolation='nearest', cmap='gray')
    plt.show()


def create_neural(data, neurons, it, accuracy, index=201):
    neural = Neural(data, neurons)
    current_image = data[index + 2000]
    label = chr(int(current_image[0] + UNICODE_OFFSET))
    neural.W1, neural.b1, neural.W2, neural.b2 = neural.gradient_descent(neural.x_train, neural.y_train, 0.10, it)
    save_neural(neural)
    # load_neural(neural)
    cnt = 0
    for i in range(2000):
        prediction = neural.make_predictions(neural.x_dev[:, i, None])
        if prediction != neural.y_dev[i]:
            # print(prediction, neural.y_dev[i])
            cnt += 1
            # print(i)
    prediction = neural.make_predictions(neural.x_train[:, index, None])
    # print("Prediction: ", chr(int(prediction + UNICODE_OFFSET)))
    # print("Label: ", label)
    #
    # current_image = current_image[1:169]
    # current_image = current_image.reshape((14, 12)) * 255
    # plt.imshow(current_image, interpolation='nearest', cmap='gray')
    # plt.show()

    # save_neural(neural)
    #
    # neural = Neural(data)
    # load_neural(neural)
    # prediction = neural.make_predictions(neural.x_train[:, index, None])
    # print("Prediction: ", chr(int(prediction + UNICODE_OFFSET)))
    # print("Label: ", label)

    # current_image = current_image[1:169]
    # current_image = current_image.reshape((14, 12)) * 255
    # plt.imshow(current_image, interpolation='nearest', cmap='gray')
    # plt.show()
    accuracy.append((1 - cnt / 2000) * 100)


def save_neural(neural):
    np.save('W1', neural.W1)
    np.save('W2', neural.W2)
    np.save('b1', neural.b1)
    np.save('b2', neural.b2)


def load_neural(neural):
    neural.W1 = np.load('W1.npy')
    neural.W2 = np.load('W2.npy')
    neural.b1 = np.load('b1.npy')
    neural.b2 = np.load('b2.npy')

# UNICODE - это моя жизнь (44F=Я, 450=???, 451=Ё)
if __name__ == '__main__':
    start_time = time.time()

    data = preparation_data()

    threads = list()
    neurons = [5, 10, 25, 35, 50]
    for neuron in neurons:
        accuracy = list()
        iterations = list()
        for i in range(25, 401, 25):
            x = threading.Thread(target=create_neural, args=(data, neuron, i, accuracy))
            threads.append(x)
            x.start()
            iterations.append(i)
        for j in range(len(threads)):
            threads[j].join()
        plt.plot(iterations, accuracy, label=f"{neuron} neurons")
    elapsed_time = round(time.time() - start_time, 2)
    print("Time elapsed: " + str(int(elapsed_time // 60)) + " minutes and "
          + str(int(elapsed_time - 60 * (elapsed_time // 60))) + " seconds")

    plt.legend()
    plt.show()
