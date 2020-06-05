import random
import tensorflow as tf
class NeuralLayer:
    '''
    Class for defining a layer of nuerons that can have multiple inputs and neurons/outputs - a neuron can only
    have one output, but it may server as inputs to many neurons in the next layer.  The equations used in this
    class assume that the bias term is included in the weights vector and that the input to the weight is 1.0.
    The equations assume that the loss function is the sum of the squares of the error where error is defined as
    the difference between a known target value and the output of the Psi function.  The Psi function is 
    1/(1+e^-z).  And z, also known as net, is the sum of product of the weights and inputs (which includes the bias).
    '''
    def __init__(self, numberOfInputs, numberOfOutputs, learningFactor=1.0):
        '''
        Nueron constructor - uses tensors - last weight is the bias term
        '''
        self.numberOfInputs = numberOfInputs
        self.numberOfNeurons = numberOfOutputs
        self.backPropagatedErrorNotSet = True
        self.learningFactor = learningFactor
        initialWeights = []
        for i in range((numberOfInputs+1)*numberOfOutputs):
            initialWeights.append(random.random() - 0.5)
        self.weights = tf.reshape(tf.convert_to_tensor(initialWeights), [numberOfInputs+1, numberOfOutputs])
        #self.weights = tf.reshape(tf.convert_to_tensor([0.0] * (numberOfInputs+1) * numberOfOutputs, tf.float32), [numberOfInputs+1, numberOfOutputs])
        #self.error = [0.0] * numberOfOutputs
        self.error = [0.0] * numberOfInputs

    def calculateOutput(self, inputs):
        '''
        Given the inputs, calculate the outputs
        '''
        self.inputs = tf.concat([inputs, [1.0]], 0) 
        self.outputs = self.psi(self.netAKAz())
        return self.outputs

    def netAKAz(self):
        '''
        Calculate the sum of the product of the weights and the inputs and add to the bia - this is net AKA z
        '''
        return tf.tensordot(self.inputs, self.weights, 1)

    def psi(self, z):
        '''
        Apply the logistic function, ψ, to the outputs
        '''
        return 1.0 / (1.0 + tf.exp(-z))

    def netWRTWeight (self, index):
        '''
        ∂zᵢ/∂wᵢ = inputᵢ  -- the change in neuron output with respect to a weight
        '''
        return self.inputs[index]

    def netWRTWeightVector (self):
        '''
        ∂zᵢ/∂wᵢ = inputᵢ  -- the change in neuron output with respect to a weight - this is a vector
        '''
        return self.inputs

    def psiWRTz(self, index):
        '''
        ∂ψᵢ/∂zᵢ = ψᵢ*(1-ψᵢ) where o = 1 / (1 + e^(-z)) -- the partial change of ψ with respect to z - this is a scalar - must designate output index
        '''
        return self.outputs[index]*(1 - self.outputs[index])

    def errorWRTPsi(self, targetArray, index):
        '''
        ∂Eᵢ/∂ψᵢ =  -(targetOutput - oᵢ)  # assuming that E is square of the error and ignoring the gain (2) - this is a scalar must designate output index
        '''
        if (self.backPropagatedErrorNotSet):
            targetOutput = targetArray[index]
            self.error[index] = -(targetOutput - self.outputs[index])
        else:
            pass  # should have been set by a higher layer
        return self.error[index]

    def updateWeights(self, target):
        '''
        Update the weights to minimize the loss
        '''
        for neuron in range(self.numberOfNeurons):
            #print("Working on neuron index {}".format(neuron))
            if neuron == 0:
                deltas = [self.errorWRTPsi(target, neuron) * self.psiWRTz(neuron) * self.netWRTWeightVector()]  # make a 1 by n vector
                #print("deltas: {}".format(deltas))
            else:
                deltas = tf.concat([deltas, [self.errorWRTPsi(target, neuron) * self.psiWRTz(neuron) * self.netWRTWeightVector()]], 0)  # tack on a new row
        self.propogateError()  # do this before updating weights
        self.weights -= tf.transpose(deltas)
        #print("final deltas: {}".format(tf.transpose(deltas)))
        #print("new weights: {}".format(self.weights))

    def propogateError(self):
        '''
        Determine error to send to previous layers
        For each neuron, determine the amount of error at it's output that needs to be applied to the input
        which is the output of the previous level.  Those individual neuron amounts then need to be summed
        across all neurons. 
        '''
        previousLayerNeuronError = [0.0] * (self.numberOfInputs + 1)
        for thisLayerNeuron in range(self.numberOfNeurons):
            error = self.error[thisLayerNeuron]
            amountForEachPreviousLayerNeuron = error * self.weights[:, thisLayerNeuron]
            previousLayerNeuronError += amountForEachPreviousLayerNeuron
        #vectorError = tf.concat([self.error, [0.0]], 0)
        #print("Error at this level:\n {}".format(vectorError))
        #print("Weights being used for propogation:\n {}".format(self.weights))
        #print("Propogated Error:\n {}".format(tf.tensordot(vectorError, self.weights, 1)))
        #self.errorForNextLayer =  tf.tensordot(vectorError, self.weights, 1)
        self.errorForNextLayer = previousLayerNeuronError
        #print("Propogated Error:\n {}".format(self.errorForNextLayer))

    def setPropogationError(self, error):
        '''
        From a higher layer, set the error propogating back to this layer
        '''
        self.error = error
        self.backPropagatedErrorNotSet = False
        
def main():
    '''
    Tests for the class NeuralLayer
    '''
    twoByTwo = NeuralLayer(2, 2, 0.05)  #  make a layer with two neurons/inputs and two outputs
    twoByTwo.calculateOutput([1.0, 1.0])
    print("neuron outputs after this pass should be [0.5, 0.5] because e⁰ = 1", twoByTwo.outputs)
    print("∂Eᵢ/∂w for neuron 0 {}".format(twoByTwo.errorWRTPsi([0.01, 0.99], 0)*twoByTwo.psiWRTz(0)* twoByTwo.netWRTWeightVector()))
    print("∂Eᵢ/∂w for neuron 1 {}".format(twoByTwo.errorWRTPsi([0.01, 0.99], 1)*twoByTwo.psiWRTz(1)* twoByTwo.netWRTWeightVector()))
    print("Initial weights:\n {}".format(twoByTwo.weights))
    twoByTwo.updateWeights([0.01, 0.99])
    # do another iteration
    for count in range(10000):
        twoByTwo.calculateOutput([1.0, 1.0])
        twoByTwo.updateWeights([0.01, 0.99])
    print("final weights:\n {}".format(twoByTwo.weights))
    print("final outputs:\n {}".format(twoByTwo.outputs))
    twoByTwo.propogateError()
    print("Propogated Error:\n {}".format(twoByTwo.errorForNextLayer))
if __name__ == '__main__':
    main()
