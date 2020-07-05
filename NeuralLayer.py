import pickle
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
    def __init__(self, numberOfInputs, numberOfOutputs, learningFactor, id="me", debug=False, filePath=None):
        '''
        Nueron constructor - uses tensors - last weight is the bias term
        '''
        self.id = id
        self.debug = debug
        self.numberOfInputs = numberOfInputs
        self.numberOfNeurons = numberOfOutputs
        self.backPropagatedErrorNotSet = True
        self.learningFactor = learningFactor
        self.normalizer = 2.0
        self.delta = [0.0] * self.numberOfNeurons
        self.weights = tf.random.uniform([numberOfInputs+1, numberOfOutputs], minval=-0.5, maxval=0.5,
                                         dtype=tf.dtypes.float32)
        self.error = [0.0] * numberOfOutputs
        self.filePath = filePath
        if filePath is not None:
            try:
                fileHandle = open(filePath, "rb")
                self.weights = pickle.load(fileHandle)
                fileHandle.close()
            except FileNotFoundError:
                pass

    def storeLayer(self, filePath):
        '''
        Store the weights that have been trained
        '''
        fileHandle = open(filePath, "wb")
        pickle.dump(self.weights, fileHandle)
        fileHandle.close()

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

    def netWRTWeight(self, index):
        '''
        ∂zᵢ/∂wᵢ = inputᵢ  -- the change in neuron output with respect to a weight
        '''
        return self.inputs[index]

    def netWRTWeightVector(self):
        '''
        ∂zᵢ/∂wᵢ = inputᵢ  -- the change in neuron output with respect to a weight - this is a vector
        '''
        return self.inputs

    def psiWRTz(self, index):
        '''
        ∂ψᵢ/∂zᵢ = ψᵢ*(1-ψᵢ) where ψ = 1 / (1 + e^(-z)) -- the partial change of ψ with respect to z - this
        is a scalar - must designate output index
        '''
        return self.outputs[index]*(1 - self.outputs[index])

    def errorWRTPsi(self, targetArray, index):
        '''
        ∂Eᵢ/∂ψᵢ =  -(targetOutput - ψᵢ)  # assuming that E is square of the error and ignoring the gain (2) -
        this is a scalar must designate output index
        '''
        if (self.backPropagatedErrorNotSet):
            targetOutput = targetArray[index]
            self.error[index] = - (self.normalizer * (targetOutput - self.outputs[index]))
        else:
            pass  # should have been set by a higher layer
        return self.error[index]

    def updateWeights(self, target=None, deltas=None):
        '''
        Update the weights to minimize the loss - if in batch mode, the deltas have been accumulated by updateDeltas
        '''
        if deltas is None:
            deltas = self.updateDeltas(target)
        self.weights -= self.learningFactor * tf.transpose(deltas)

    def updateDeltas(self, target, deltas=None):
        '''
        Update the deltas during batch processing
        '''
        for neuron in range(self.numberOfNeurons):
            if neuron == 0:
                deltaDeltas = tf.reshape(tf.convert_to_tensor(self.errorWRTPsi(target, neuron)
                                                                  * self.psiWRTz(neuron)
                                                                  * self.netWRTWeightVector()),
                                             [1, len(self.netWRTWeightVector())])  # make a 1 by n vector
            else:
                deltaDeltas = tf.concat((deltaDeltas, [self.errorWRTPsi(target, neuron)
                                                           * self.psiWRTz(neuron)
                                                           * self.netWRTWeightVector()]), 0)  # tack on a new row
            if self.debug:
                print("updateDeltas - layer {}, neuron {}, weight deltaDeltas\n{}".
                          format(self.id, neuron, deltaDeltas))

        if deltas is None:
            deltas = deltaDeltas
        else:
            deltas += deltaDeltas
        self.propagateError()  # do this before updating weights
        return deltas

    def propagateError(self):
        '''
        Determine error to send to previous layers
        For each neuron, determine the amount of error at it's output that needs to be applied to the input
        which is the output of the previous level.  Those individual neuron amounts then need to be summed
        across all neurons.
        '''
        previousLayerNeuronError = [0.0] * (self.numberOfInputs + 1)
        for thisLayerNeuron in range(self.numberOfNeurons):
            error = self.error[thisLayerNeuron]
            amountForEachPreviousLayerNeuron = error * self.weights[:, thisLayerNeuron] * self.psiWRTz(thisLayerNeuron)
            if self.debug:
                print("sum of weights for neurons at this layer: {}".
                      format(tf.reduce_sum(self.weights[:, thisLayerNeuron])))
                print("propagateError - in layer {}, neuron {}, contribution:{}".
                      format(self.id, thisLayerNeuron, amountForEachPreviousLayerNeuron))
                print("propagateError - Error {}, weights {}".format(error, self.weights[:, thisLayerNeuron]))
            previousLayerNeuronError += amountForEachPreviousLayerNeuron
        self.errorForNextLayer = previousLayerNeuronError
        if self.debug:
            print("propagateError - in layer {}, the next layer's error will be\n {}".
                  format(self.id, previousLayerNeuronError))

    def setPropagationError(self, error):
        '''
        From a higher layer, set the error propogating back to this layer
        '''
        self.error = error
        if self.debug:
            print("setPropagationError - setting propagation error in layer {} to\n {}".
                  format(self.id, self.error))
        self.backPropagatedErrorNotSet = False

    def setLearningFactor(self, factor):
        '''
        Setter for learning factor
        '''
        self.learningFactor = factor

