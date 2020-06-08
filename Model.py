import NeuralLayer as nl
import tensorflow as tf
class Model:
    '''
    Class for defining a model that consists of nueral layers.  The first layer is always the inputs, which means that it does not exist as a NeuralLayer.
    All subsequent layers are completely interconnected except for the final layer
    '''
    def __init__(self, inputOutputList):
        '''
        Model constructor - contrusts layers from the list entries
        '''
        self.layers = []
        for (inputs, outputs, learningFactor) in inputOutputList:
            self.layers.append(nl.NeuralLayer(inputs, outputs, learningFactor))

    def feedForward(self, inputs):
        '''
        Given the inputs, propagate them through the model layers
        '''
        layerOutputs = inputs
        for aLayer in self.layers:
            layerOutputs = aLayer.calculateOutput(layerOutputs)
        return layerOutputs

    def updateDeltas(self, target, deltas=None):
        '''
        Update the deltas in all the layers
        '''
        reversedLayers = self.layers.copy()
        reversedLayers.reverse()
        lastLayer = len(reversedLayers) - 1
        newDeltaList = False
        if deltas is None:
            deltas = []  #  make a list of deltas, one for each layer
            newDeltaList = True
        for index, layer in enumerate(reversedLayers):
            #print("Working on layer index {}".format(index))
            #print("updating deltas for Index: {}, layer: {}".format(index, lastLayer - index))
            if newDeltaList:
                deltas.append(layer.updateDeltas(target))
            else:
                deltas[index] = layer.updateDeltas(target, deltas=deltas[index])
            if index < lastLayer:
                reversedLayers[index+1].setPropogationError(layer.errorForNextLayer)
        #print("returning deltas:\n {}".format(deltas))
        return deltas
        
    def updateWeights(self, target=None, deltas=None):
        '''
        Update the weights and propogte the error of all layers
        '''
        reversedLayers = self.layers.copy()
        reversedLayers.reverse()
        lastLayer = len(reversedLayers) - 1
        for index, layer in enumerate(reversedLayers):
            #print("Working on layer index {}".format(index))
            if deltas is None:
                layer.updateWeights(target)
                if index < lastLayer:
                    reversedLayers[index+1].setPropogationError(layer.errorForNextLayer)
            else:
                #print("type of deltas at this point is: {}".format(type(deltas[index])))
                #print("updating weights using batch calculated deltas. Index: {}, layer: {}".format(index, lastLayer - index))
                layer.updateWeights(deltas=deltas[index])
        
def main():
    '''
    Tests for the class Model
    '''
    model = Model([(2,2,0.01), (2,2,0.01)])
    print("Initial Weights for layer 0:\n {}".format(model.layers[0].weights))
    print("Initial Weights for layer 1:\n {}".format(model.layers[1].weights))
    for iteration in range(2000):
        model.feedForward([1.0, 1.0])
        model.updateWeights([0.01, 0.99])

    print("Final Weights for layer 0:\n {}".format(model.layers[0].weights))
    print("Final Weights for layer 1:\n {}".format(model.layers[1].weights))
    print("model is predicting:\n {}".format(model.feedForward([1.0, 1.0])))  #  lets see how close we got
    
    xorModel = Model([(2,2,0.01), (2,2,0.01)])
    print("Initial Weights for layer 0:\n {}".format(xorModel.layers[0].weights))
    print("Initial Weights for layer 1:\n {}".format(xorModel.layers[1].weights))
    for iteration in range(1000):
        xorModel.feedForward([1.0, 1.0])
        xorModel.updateWeights([1.0, 0.0])
        xorModel.feedForward([0.0, 0.0])
        xorModel.updateWeights([1.0, 0.0])
        xorModel.feedForward([1.0, 0.0])
        xorModel.updateWeights([0.0, 1.0])
        xorModel.feedForward([0.0, 1.0])
        xorModel.updateWeights([0.0, 1.0])

    print("Final Weights for layer 0:\n {}".format(xorModel.layers[0].weights))
    print("Final Weights for layer 1:\n {}".format(xorModel.layers[1].weights))
    print("model is predicting:\n {}".format(xorModel.feedForward([1.0, 1.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([0.0, 0.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([1.0, 0.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([0.0, 1.0])))  #  lets see how close we got
    
if __name__ == '__main__':
    main()
