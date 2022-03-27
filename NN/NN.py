import numpy as np
import scipy.special

class NeuralNetwork:
    
    def __init(self, inputNode, hiddenNode, outputNode,learningRate):
        self.iNode = inputNode
        self.hNode = hiddenNode
        self.oNode = outputNode

        # Learning rate
        self.lr = learningRate

        # Setting the weight matrices
        self.wih = np.random.normal(0.0,pow(self.iNode,-0.5),self.hNode,self.iNode)
        self.who = np.random.normal(0.0,pow(self.hNode,-0.5),self.oNode,self.hNode)

        # Sigmoid Activation Function
        self.activationFunction = lambda x: scipy.special.expit(x)

        
    def train(self, input_list, target_list):

        inputs = np.array(input_array,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T

        # Calculating hidden node's input and outputs
        hiddenInputs = np.dot(self.wih,inputs)
        hiddenOutputs = self.activationFunction(self.hiddenInputs)

        # Calculating output node's input and outputs
        final_input = np.dot(self.who,hiddenOutputs)
        final_output = self.activationFunction(self.final_input)

        # Calculating the Error (target - output)
        output_error = targets - final_output

        # hidden layer error is the output_errors, split by weights, 
        # recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error)

        # Update the hidden weight for link between the hidden node and output node
        self.who += self.lr * np.dot(output_error * final_output * (1 - final_output),np.tanspose(hiddenOutputs)) 


        # Update the hidden weight for link between the input node and hidden node
        self.wih += self.lr *np.dot(hidden_error * hiddenOutputs * (1 - hiddenOutputs),np.tanspose(inputs))

    def query(self, input_list, target_list):

        # Converting input list and target list into 2D array
        inputs = np.array(input_array,ndmin=2).T

        # Calculating hidden node's input and outputs
        hiddenInputs = np.dot(self.wih,inputs)
        hiddenOutputs = self.activationFunction(self.hiddenInputs)

        # Calculating output node's input and outputs
        final_input = np.dot(self.who,hiddenOutputs)
        final_output = self.activationFunction(self.final_input)

        return final_output
