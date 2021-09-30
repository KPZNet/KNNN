import copy
from datetime import *
import matplotlib.pyplot as plt
from KKeras import *

#Print error rate over epochs to see training trend
def plot_error(epocs, error_history) :
    plt.plot ( epocs, error_history, label="errors" )
    plt.title ( "Training" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "error" )
    plt.legend ()
    plt.show ()

#Plot weight updates to see training trends over epochs
def plot_weights(epocs, weight_history) :
    wh = np.array ( weight_history )
    s = wh.shape[1]
    [plt.plot ( epocs, wh[:, i], label="weight {0}".format ( i ) ) for i in range ( s )]
    plt.title ( "Weights" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "weight" )
    plt.legend ()
    plt.show ()

#Sigmoid function
def sigmoidA(x) :
    return 1 / (1 + np.exp ( -x ))

#Sigmoid function derivative
def sigmoidA_derivative(x) :
    return x * (1 - x)

#Neural net class
#Uses replaceable activation functions
class NeuralNetwork :
    def __init__(self) :
        self.error_history = []
        self.epoch_list = []
        self.weight_history = []
        self.stop_delta = 0.001

    def train(self, training_input, training_output, epochs=1000) :
        self.weights = np.array ( np.random.normal ( size=(col, 1) ) )
        for epoch in range ( epochs ) :
            stop = self.run_epoch ( training_input, training_output, epoch )
            if stop :
                break

    def run_epoch(self, training_input, training_output, epoch) :
        stop = False
        self.feed_forward ( training_input, sigmoid_fn=sigmoidA )
        self.backpropagation ( training_input, training_output, sigmoid_fn_derivative=sigmoidA_derivative )

        err = np.average ( np.abs ( self.error ) )
        if err < self.stop_delta :
            stop = True
        self.error_history.append ( err )
        self.epoch_list.append ( epoch )
        return stop

    def feed_forward(self, training_input, sigmoid_fn) :
        self.hidden = sigmoid_fn ( np.dot ( training_input, self.weights ) )

    def backpropagation(self, training_input, training_output, sigmoid_fn_derivative) :
        self.error = training_output - self.hidden
        delta = self.error * sigmoid_fn_derivative ( self.hidden )
        self.weights += np.dot ( training_input.T, delta )
        self.weight_history.append ( copy.deepcopy ( self.weights ) )

    def predict(self, sigmoid_fn, new_input) :
        prediction = sigmoid_fn ( np.dot ( new_input, self.weights ) )
        return prediction

def test_run_random(nnet, num, row, col):
    inputs = np.random.randint ( 2, size=(row, col) )
    print("\nTEST RESULTS")
    for input in inputs:
        e = input[2]
        print ( 'Input: ', input, 'Expected:', e, ' , NN Result: ', nnet.predict ( sigmoid_fn=sigmoidA, new_input=input ) )

np.random.seed(datetime.now().microsecond)
#np.random.seed ( 5632 )
row, col = 100, 8

inputsA = np.random.randint ( 2, size=(row, col) )
outputsA = np.array ( [inputsA[:, 2]] ).T

NNN = NeuralNetwork ()
NNN.train ( inputsA, outputsA )

plot_error ( NNN.epoch_list, NNN.error_history )
plot_weights ( NNN.epoch_list, NNN.weight_history )

run_test_1 = np.array ( [[1, 1, 1, 0, 0, 1, 0, 1]] )
run_test_2 = np.array ( [[0, 0, 0, 1, 0, 1, 1, 0]] )

test_run_random(NNN, 10, 5, col)

#Keras_run ( inputsA, outputsA, run_test_1, run_test_2 )
#test_net ( NNN, run_test_1, run_test_2 )