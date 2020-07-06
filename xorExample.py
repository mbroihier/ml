import getopt
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import Model

def main(bypass, cycles):
    '''
    Analysis of XOR learning operation
    '''
    if cycles is None:
        cycles = 1250
    tf.random.set_seed(1671)
    # Create a model with two neural layers.  The first layer, "0", has two inputs and three neurons/outputs with a learning factor/rate of 0.5.
    # The second layer, "1", has 3 inputs and one neuron/output with a learning factor/rate of 0.5.  The model is stored at xorModel0 and
    # xorModel1 in the current directory.  If it exists, the weights stored in the file are used as the initial weights.
    xorModel = Model.Model([(2, 3, 0.5), (3, 1, 0.5)], filePath="xorModel")
    print("Initial Weights for layer 0:\n {}".format(xorModel.layers[0].weights))
    print("Initial Weights for layer 1:\n {}".format(xorModel.layers[1].weights))
    oo = []  # Stores the output of response when input is 0 0
    ol = []  #                     //                      0 1
    lo = []  #                     //                      1 0
    ll = []  #                     //                      1 1
    layer0Error = []
    layer1Error = []
    for iteration in range(cycles):
        result = xorModel.feedForward([1.0, 1.0])  # input is 1 1 so the expected output is 0
        xorModel.updateWeights([0.0])
        if not bypass:
            ll.append(result)
            layer0Error.append(xorModel.layers[0].error.numpy())
            layer1Error.append(xorModel.layers[1].error.copy())
        result = xorModel.feedForward([1.0, 0.0])  # input is 1 0 so the expected output is 1
        xorModel.updateWeights([1.0])
        if not bypass:
            lo.append(result)
            layer0Error.append(xorModel.layers[0].error.numpy())
            layer1Error.append(xorModel.layers[1].error.copy())
        result = xorModel.feedForward([0.0, 1.0])  # input is 0 1 so the expected output is 1
        xorModel.updateWeights([ 1.0])
        if not bypass:
            ol.append(result)
            layer0Error.append(xorModel.layers[0].error.numpy())
            layer1Error.append(xorModel.layers[1].error.copy())
        result = xorModel.feedForward([0.0, 0.0])  # input is 0 0 so the expected output is 0
        xorModel.updateWeights([0.0])
        if not bypass:
            oo.append(result)
            layer0Error.append(xorModel.layers[0].error.numpy())
            layer1Error.append(xorModel.layers[1].error.copy())
        
    print("model is predicting:\n {}".format(xorModel.feedForward([1.0, 1.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([0.0, 0.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([1.0, 0.0])))  #  lets see how close we got
    print("model is predicting:\n {}".format(xorModel.feedForward([0.0, 1.0])))  #  lets see how close we got
    print("Final Weights for layer 0:\n {}".format(xorModel.layers[0].weights))
    print("Final Weights for layer 1:\n {}".format(xorModel.layers[1].weights))
    xorModel.storeModel("xorModel")  # store the current results of the model
    if bypass:
        sys.exit(0)
    fig = plt.figure()
    plt.plot(oo)
    plt.plot(ol, "*")
    plt.plot(lo, "^")
    plt.plot(ll, "+")
    plt.title("Output Each Cycle")
    plt.xlabel("Cycle")
    plt.ylabel("Value")
    plt.savefig("LearningXOR.png")
    plt.show()

    fig = plt.figure()
    plt.plot(layer0Error)
    plt.title("Layer 0 Error Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()
    fig = plt.figure()
    plt.plot(layer1Error)
    plt.title("Layer 1 Error Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

    
if __name__ == '__main__':
    bypass = False
    cycles = None
    usage = "python3 xorExample.py [-c <number of cycles>] [--bypass]\n   bypass - bypass graphic display\n   -c - learning cycles to perform"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ["bypass"])
        # print("opts:  {}, args: {}".format(opts, args))
        for opt, arg in opts:
            if opt == "-c":
                print("Setting cycles to {}".format(arg))
                cycles = int(arg)
            else:
                if opt == "--bypass":
                    bypass = True
                else:
                    print("options given were: {} {}".format(opt, arg))
                    print(usage)
                    sys.exit(-1)
        for arg in args:
            print("invalid argument observed: {}".format(arg))
            print(usage)
            sys.exit(-1)
    except getopt.GetoptError:
        print("GetoptError exception")
        print(usage)
        sys.exit(-1)
    main(bypass, cycles)
