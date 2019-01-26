#!/usr/bin/env python3

"""
Some utilities functions
"""

import numpy as np
import tree

# Maths functions

def softmax(w):
    """
    Straightforward implementation of softmax function
    """
    e = np.exp(w)
    dist = e / np.sum(e)
    
    return dist

def actFct(x):
    """
    The NN activation function (here tanh)
    """
    return np.tanh(x)

def actFctDerFromOutput(x):
    """
    Derivate of the activation function
    WARNING: In this version, we take as input an output value 
    after the activation function (x = tanh(output of the tensor)).
    This works because the derivate of tanh is function of tanh
    """
    return 1.0 - x**2

#def actFctDer(x):
    #"""
    #Derivate of the activation function
    #"""
    #return 1.0 - np.tanh(x)**2

# Other utils functions
    
def loadDataset(filename):
    """
    Load and return the dataset given in parameter
    """

    dataFile = open(filename, 'r')
    lines = dataFile.readlines()
    dataFile.close()

    dataset = []

    # Extract rows
    for line in lines:
        dataset.append(tree.Tree(line)) # Create the tree for each sentence
    return dataset

def pltCMatrix(cMatrix,TrainOrTest):
    import matplotlib.pyplot as plt
    import numpy as np
    path="""F:\\Local Disk(D)\\video\\Codes\M.sc Thesis\\RNTN-code\\english-RNTN_Original\\save\\"""
    #Normalize Confusion matrix
    cMatrix = cMatrix.astype('float') / cMatrix.sum(axis=1)[:, np.newaxis]
    #Plot normalize confusion matrix
    plt.matshow(cMatrix)
    plt.title(str(TrainOrTest))
    plt.colorbar()
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cmap=plt.cm.binary
    plt.imshow(cMatrix, interpolation='nearest', cmap=cmap)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, format(cMatrix[i, j], '.1f'))
    #plt.savefig(path + str(TrainOrTest) + "_cMat.png")
    plt.show()
    print('Saving cMat Done!')
    #plt.show()











