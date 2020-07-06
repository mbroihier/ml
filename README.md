# Machine Learning Python Classes

This repository contains classes I wrote inspired by https://github.com/zotroneneis/machine_learning_basics, https://github.com/mattm/simple-neural-network, and https://github.com/stephencwelch/Neural-Networks-Demystified.

There are two Python modules that define two classes: NeuralLayer and Model.  A neural layer can be constructed with the neural layer class.  The layer can have an arbitrary number of inputs and neurons.  All inputs are connected to all neurons.  The number of outputs of the layer is equal to the number of neurons.  The NeuralLayer class isn't typically used directly.  A model creates and contains the neural layers that will be in a model instance.  A model can be contructed with the Model class in the Model module.  A model is a set of neural layers that will accept a fixed number of inputs and produce a fixed number of outputs.  The number of neuron layers is arbitrary, but all layers are "densely" connected.  That is, the outputs of lower layers are the inputs to their adjacent next layer.  So if the first layer has 15 inputs and 3 neurons, the next layer must have 3 inputs, but can have an arbitrary number of neurons.

Model objects have a feedForward method that, given an input, produce a model output.  They also have updateDeltas and updateWeights methods that accept expected output ("truth") and produce deltas that can be used to update neural layer weights.  updateDeltas is intended to be used by the user to process batches of inputs.  After processing a batch, a call to updateWeights passing the accumulated deltas updates the weights.  If the user wants to update the weights with each training sample, updateWeights will accept the expected output, produce the deltas, and update the weights all in one call.

A model can be stored by calling its storeModel method.  The model can be restored to its current state by instantiating a new model with the same neural layers and the path name to the model that was used by storeModel.

A Jupyter notebook is included that compares how well the back propagation gradient matches a calcuated gradient.  This can be altered to examine other models.  As long as the model's gradient and calculated gradient have a ratio close to one, the model is sane.

A model that learns XOR is included as an example of using the classes.  The figure below illustrates the change in its outputs as it learns to recognize the appropriate response to binary combinations.
[!alt text](LearningXOR.pdf)

# Setup

I'm running this on a Raspberry PI 4 under Raspbian (buster).  The following is the setup I did, but it's possible that it is missing a bit since I didn't install this from scratch.  Let me know if you find that I'm missing something.

From a bash shell:

1) pip3 install virtualenv
2) virtualenv env (can be any name)
3) source env/bin/activate
4) wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.2.0/tensorflow-2.2.0-cp37-none-linux_armv7l.whl
5) pip3 install tensorflow-2.2.0-cp37-none-linux_armv7l.whl 
6) pip3 install jupyter
7) pip3 install pandas
8) pip3 install seaborn
9) pip3 install celluloid
10) pip3 install IPython
11) pip3 install matplotlib
12) pip3 install pycodestyle pycodestyle_magic
13) pip3 install pyflakes
14) pip3 install flake8
15) jupyter notebook --generate-config

To run the XOR example (while in the "env" environment):
```
python3 xorExample.py
```