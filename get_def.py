from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


# MNIST 함수 호출
def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    L, H, W = x_train.shape
    x_train = x_train.reshape(-1, H*W)
    x_test = x_test.reshape(-1, H*W)

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


def show_loss(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='upper left')
    return plt.show()

def show_acc(model):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'], loc='upper left')
    return plt.show()


# cifar10 데이터셋

def get_cifar_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    L, H, W, C = np.shape(x_train)

    x_train = x_train.reshape(-1, H*W*C)
    x_test = x_test.reshape(-1, H*W*C)

    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)