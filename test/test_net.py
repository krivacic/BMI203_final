from nn import net
import numpy as np
import os

def test_network_autoencoder():
    # Tests the network's ability to trian autoencode the identity matrix.

    Xpos = np.array([[1,1,1,1,1,1,1,1]])
    Xneg = np.array([[0,0,0,0,0,0,0,0]])

    ypos = np.ones((Xpos.shape))
    yneg = np.zeros((Xneg.shape))

    X = np.concatenate((Xpos,Xneg))
    y = np.concatenate((ypos,yneg))

    Xtestpos = ([[1,1,1,1,1,1,1,1]])
    Xtestneg = ([[0,0,0,0,0,0,0,0]])

    outlist1 = []
    outlist2 = []
    NN = net.neural_network(4,1,8)

    for i in range(0,5):
        NN.__init__(8,8,3)
        NN.train(X,y,10000,20,0)
        out1 = NN.forward(Xtestpos)
        outlist1.append(out1)
        out2 = NN.forward(Xtestneg)
        outlist2.append(out2)

    assert np.average(outlist1) > 0.9 and np.average(outlist2) < 0.1

def test_network_stochastic():
    # Trains the neural network on a simple dataset using stochastic gradient descent and tests it on simple example data.

    # Positive data all have a "1" in the third position. Negative data all has a "0" in the third position.
    # This test makes sure the net learns to heavily weigh the third position but ignores the others.
    Xpos = np.array([[0,0,1,0],[1,0,1,0],[0,1,1,0],[0,0,1,1],[0,0,1,0],[0,0,1,0],[1,1,1,0],[1,1,1,1]])
    Xneg = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,1]])

    ypos = np.ones((Xpos.shape[0],1))
    yneg = np.zeros((Xneg.shape[0],1))

    X = np.concatenate((Xpos,Xneg))
    y = np.concatenate((ypos,yneg))

    Xtestpos = ([[0,0,1,0]])
    Xtestneg = ([[1,1,0,1]])

    outlist1 = []
    outlist2 = []
    NN = net.neural_network(4,1,8)

    for i in range(0,5):
        NN.__init__(4,1,8)
        NN.train_stochastic(X,y,10000,20,0,Xpos.shape[0],Xneg.shape[0])
        out1 = NN.forward(Xtestpos)
        outlist1.append(out1)
        out2 = NN.forward(Xtestneg)
        outlist2.append(out2)

    assert np.average(outlist1) > np.average(outlist2)

def test_network_batch():
    # Trains the neural network on a simple dataset and tests it on another, this time using batch descent.

    # Positive data all have a "1" in the third position. Negative data all has a "0" in the third position.
    # This test makes sure the net learns to heavily weigh the third position but ignores the others.
    Xpos = np.array([[0,0,1,0],[1,0,1,0],[0,1,1,0],[0,0,1,1],[0,0,1,0],[0,0,1,0],[1,1,1,0],[1,1,1,1]])
    Xneg = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,1]])

    ypos = np.ones((Xpos.shape[0],1))
    yneg = np.zeros((Xneg.shape[0],1))

    X = np.concatenate((Xpos,Xneg))
    y = np.concatenate((ypos,yneg))

    Xtestpos = ([[0,0,1,0]])
    Xtestneg = ([[1,1,0,1]])

    outlist1 = []
    outlist2 = []
    NN = net.neural_network(4,1,8)

    for i in range(0,5):
        NN.__init__(4,1,8)
        NN.train(X,y,10000,20,0)
        out1 = NN.forward(Xtestpos)
        outlist1.append(out1)
        out2 = NN.forward(Xtestneg)
        outlist2.append(out2)

    assert np.average(outlist1) > np.average(outlist2)
