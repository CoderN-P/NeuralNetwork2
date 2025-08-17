from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from . import NeuralNetwork, ReLu, Sigmoid, Softmax, CrossEntropy, Dense

def train_and_evaluate_handwriting_detection():
    """
    Train and evaluate a handwriting detection model using the MNIST dataset.
    """

        # Load the dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    
    X = X / 255.0  # Normalize pixel values to [0, 1]
    # Convert labels to integers
    y = y.astype(np.int64)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    X_train = X_train.reshape(*X_train.shape, 1)
    y_train = y_train.reshape(*y_train.shape, 1,  1)
    X_test = X_test.reshape(*X_test.shape, 1)
    y_test = y_test.reshape(*y_test.shape, 1, 1)
    
    network = NeuralNetwork([
        Dense(784, 128),
        Sigmoid(),
        Dense(128, 64),
        Sigmoid(),
        Dense(64, 10),
        Softmax()
    ], CrossEntropy())
    
    
    network.train(X_train, y_train, 10, 0.01)
    print(network.predict(X_test[0]))
    network.save("handwriting_detection_model.json")
    
    preds = [np.argmax(network.predict(image)) for image in X_test]
    preds = np.array(preds)
    
    y_test = y_test.reshape(len(y_test))
    
    correct = [pred == real for pred, real in zip(preds, y_test)]
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, preds):
        ax.set_axis_off()
        image = image.reshape(28, 28)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()
    
    
    print(sum(correct)/len(y_test))
    
    
def evaluate_handwriting_detection(path):
    network = NeuralNetwork.load(path)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    
    X = X / 255.0  # Normalize pixel values to [0, 1]
    # Convert labels to integers
    y = y.astype(np.int64)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    X_train = X_train.reshape(*X_train.shape, 1)
    y_train = y_train.reshape(*y_train.shape, 1,  1)
    X_test = X_test.reshape(*X_test.shape, 1)
    y_test = y_test.reshape(*y_test.shape, 1, 1)

    preds = [np.argmax(network.predict(image)) for image in X_test]
    preds = np.array(preds)

    y_test = y_test.reshape(len(y_test))

    correct = [pred == real for pred, real in zip(preds, y_test)]
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, preds):
        ax.set_axis_off()
        image = image.reshape(28, 28)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()


    print(sum(correct)/len(y_test))


