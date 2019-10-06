# Arora, Aman
# 1001-667-638
# 2019-09-22
# Assignment-01-01

import numpy as np
import itertools


class Perceptron(object):
    def __init__(self, input_dimensions=2, number_of_classes=4, seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes = number_of_classes
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights, initialize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions + 1)

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initialize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1))

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        # Adding a row of 1 to the input received from the training function
        # Taken boundary value of 0.0 to activate the activation for 0 or 1
        # Took the dot product of the weights in the loop and input received from train function and cal_error function
        # Converted to 0 or 1 based on the output greater than equal to or less than boundary value and returns
        # the output in a temp list of array
        temp =[]
        X1 = np.insert(X, 0, 1, axis=0)
        bound_con = 0.0
        for i in range(len(self.weights)):
            dot_pro = np.dot(self.weights[i], X1)
            exp_out = np.where(dot_pro >= bound_con, 1, 0)
            temp.append(exp_out)
        return temp

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        """raise Warning("You must implement print_weights")"""

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param Y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        # running the loop for specific number of epochs
        # for each epochs running the loop to extract each sample from input
        # calculating the error by subtracting the predicted output from the expected output
        # Further calculating the updating parameter and updating the weights for each run
        X1 = np.insert(X, 0, 1, axis=0)
        for _ in range(num_epochs):
            for i in range(X1.shape[1]):
                exp_output = np.resize(np.array(Y[:, i]), (self.number_of_classes, 1))
                error = np.array(exp_output - np.resize(self.predict(np.transpose(X)[i]), (self.number_of_classes, 1)))
                error = np.resize(error, (self.number_of_classes, 1))
                update_parm = alpha * error * np.transpose(X1)[i]
                self.weights = self.weights + (update_parm)

    def calculate_percent_error(self, X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param Y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        prediction = np.array(self.predict(X))
        error = np.array(Y - prediction)
        count = 0
        for i in range(error.shape[1]):
            if any(error[:, i]) !=0:
                count += 1
        return count/X.shape[1]


if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    """X_train = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
                        [1, 2, -2, 0, 2, 1, -1, -2]])"""
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    """Y_train = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1]])"""
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n", model.weights)
    print("****** Input samples ******\n", X_train)
    print("****** Desired Output ******\n", Y_train)
    percent_error = []
    for k in range(20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train, Y_train))
    print("******  Percent Error ******\n", percent_error)
    print("****** Model weights ******\n", model.weights)