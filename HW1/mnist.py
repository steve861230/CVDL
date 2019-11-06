import sys
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
def show():
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)


    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    nsample = 1
    randidx = np.random.randint(trainimg.shape[0], size=nsample)

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
        curr_label = np.argmax(trainlabel[i, :] ) # Label
        plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.title("" + str(i + 1) + "th Training Data " 
                                + "Label is " + str(curr_label))
    plt.show()