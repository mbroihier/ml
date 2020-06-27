import tensorflow as tf
import NeuralLayer


class Model:
    '''
    Class for defining a model that consists of neural layers.  The first layer is always the inputs, which
    means that it does not exist as a NeuralLayer.  All subsequent layers are completely interconnected
    except for the final layer.
    '''
    def __init__(self, inputOutputList, debug=False, filePath=None):
        '''
        Model constructor - contrusts layers from the list entries
        '''
        self.layers = []
        layerIndex = 0

        for entryTuple in inputOutputList:
            layerid = None
            if isinstance(entryTuple[-1], str):
                layerid = entryTuple[-1]
            inputs = entryTuple[0]
            outputs = entryTuple[1]
            learningFactor = entryTuple[2]
            if filePath is None:
                weightFilePath = filePath
            else:
                weightFilePath = filePath + str(layerIndex)
                layerIndex += 1
            self.layers.append(NeuralLayer.NeuralLayer(inputs, outputs, learningFactor, layerid, debug, weightFilePath))

    def storeModel(self, filePath):
        '''
        Store the weights for all of the layers of this model
        '''
        layerIndex = 0
        for layer in self.layers:
            layer.storeLayer(filePath + str(layerIndex))
            layerIndex += 1

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
            deltas = []  # make a list of deltas, one for each layer
            newDeltaList = True
        for index, layer in enumerate(reversedLayers):
            topIndicator = index == 0
            if newDeltaList:
                deltas.append(layer.updateDeltas(target, topIndicator))
            else:
                deltas[index] = layer.updateDeltas(target, topIndicator, deltas=deltas[index])
            if index < lastLayer:
                reversedLayers[index+1].setPropagationError(layer.errorForNextLayer)
        return deltas

    def updateWeights(self, target=None, deltas=None):
        '''
        Update the weights and propogte the error of all layers
        '''
        reversedLayers = self.layers.copy()
        reversedLayers.reverse()
        lastLayer = len(reversedLayers) - 1
        for index, layer in enumerate(reversedLayers):
            if deltas is None:
                layer.updateWeights(target)
                if index < lastLayer:
                    reversedLayers[index+1].setPropagationError(layer.errorForNextLayer)
            else:
                layer.updateWeights(deltas=deltas[index])

