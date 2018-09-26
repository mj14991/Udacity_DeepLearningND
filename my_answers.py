
import numpy as np
import pandas as pd


# Implement the sigmoid function to use as the activation function. Set self.activation_function in __init__ to your sigmoid function.
# Implement the forward pass in the train method.
# Implement the backpropagation algorithm in the train method, including calculating the output error.
# Implement the forward pass in the run method.
# Hidden layer uses sigmoid funciton, output layer > f(x) = x.


# Hint: You'll need the derivative of the output activation function ( f(x)=xf(x)=x ) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation  y=xy=x . What is the slope of that equation? That is the derivative of  f(x)f(x) . Slope 1

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # If the lambda code above is not something you're familiar with,
#         You can uncomment out the following three lines and put your
#         implementation there instead.

#         def sigmoid(x):
#            return 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation here
#         self.activation_function = sigmoid

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
#         print("Hidden Inputs:", hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
#         print ("Hidden Outputs:", hidden_outputs)

        # TODO: Output layer - Replace these values with your calculations.
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer

#         print ("hidden output weight:", self.weights_hidden_to_output)

#         print ("Final outputs:", final_outputs)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

#         print("Error:", error)

        # TODO: Calculate the hidden layer's contribution to the error

        output_error_term = error
#         print ("output_error_term:", output_error_term)

        hidden_error = np.dot(error, self.weights_hidden_to_output.T)

#         print ("hidden error:", hidden_error)

        # TODO: Backpropagated error terms - Replace these values with your calculations.

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

#         print ("hidden_error_term:", hidden_error_term)
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
#         print ("delta_weights input hidden:", delta_weights_i_h)
        # Weight step (hidden to output)
        delta_weights_h_o += error * hidden_outputs[:, None]
#         print ("delta_weights output hidden:", delta_weights_h_o)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''

        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
#         print ("self.weights_hidden_to_output", self.weights_hidden_to_output)
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
#         print ("self.weights_input_to_hidden", self.weights_hidden_to_output)

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2500
learning_rate = 0.50
hidden_nodes = 25
output_nodes = 1
