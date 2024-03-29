#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import random
import numpy as np
import train
import utils
import vocabulary

# Parameters (are arrays for the grid search)
nbEpoch = 60
miniBatchSize = [5]
adagradResetNbIter = [6] # Reset every X iterations (0 for never) << does not seem to have that much impact except the performance drop at each reset on the training set

#learningRate = [0.00001, 0.0001, 0.001, 0.01] # << Seems to be the best
learningRate = [0.01] # << Seems to be the best

#regularisationTerm = [0.00001, 0.0001, 0.001, 0.01]
regularisationTerm = [0.00001]

# Path and name where the infos will be saved
outputDir = "save/"
outputNameDefault = "train"

def main(outputName):
    print("Welcome into RNTN implementation 0.6 (recording will be on ", outputName, ")")
    
    random.seed("MetaMind") # Lucky seed ? Fixed seed for replication
    np.random.seed(7)
    
    print("Parsing dataset, creating dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab()
    
    # Loading dataset
    datasets = {}
    datasets['training'] = utils.loadDataset("trees/train.txt");
    print("Training loaded !")
    
    datasets['testing'] = utils.loadDataset("trees/test.txt");
    print("Testing loaded !")
    datasets['validating'] = utils.loadDataset("trees/dev.txt");
    print("Validation loaded !")
    
    print("Datasets loaded !")
    print("Nb of words", vocabulary.vocab.length());
    
    # Datatransform (normalisation, remove outliers,...) ?? > Not here
    
    # Grid search on our hyperparameters (too long for complete k-fold cross validation so just train/test)
    for mBS in miniBatchSize:
        for aRNI in adagradResetNbIter:
            for lR in learningRate:
                for rT in regularisationTerm:
                    params = {}
                    params["nbEpoch"]            = nbEpoch
                    params["learningRate"]       = lR
                    params["regularisationTerm"] = rT
                    params["adagradResetNbIter"] = aRNI
                    params["miniBatchSize"]      = mBS
                    # No need to reset the vocabulary values (contained in model.L so automatically reset)
                    # Same for the training and testing set (output values recomputed at each iterations)
                    model = train.train(outputName, datasets, params)

    # TODO: Plot the cross-validation curve
    # TODO: Plot a heat map of the hyperparameters cost to help tunning them ?

    ## Validate on the last computed model (Only used for final training)
    #print("Training complete, validating...")
    #vaError = model.computeError(datasets['validating'], True)
    #print("Validation error: ", vaError)

    print("The End. Thank you for using this program!")
    

if __name__ == "__main__":
    # Simple parsing to get the model name
    outputName = outputNameDefault
    if len(sys.argv) > 1:
        outputName = sys.argv[1]
    main(outputDir + outputName)
