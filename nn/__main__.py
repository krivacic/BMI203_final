from .net import neural_network
from .io import get_data, parse, read_negatives
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
import random

"""
Usage:
From parent folder, type python -m nn to run.
Options:
'-i <#>' or '--iter <#>' to number of iterations
'-a <#>' or '--alpha <#>' to set value for alpha
'-l <#>' or '--lambda <#>' to set lambda value
'-t' or '--train' to train the model on all known data, then run the unknown data through the model.
'-b' or '--batch' to determine neural net performance using batch descent (not recommended)
"""


parser = OptionParser()
parser.add_option("-i","--iter",action = "store",type = "int",dest = "iter", default = "100")
parser.add_option("-a","--alpha",action = "store",type = "float", dest = "alpha", default = "10")
parser.add_option("-l","--lambda",action = "store",type = "float",dest = "lam", default = "0")
parser.add_option("-t","--train",action = "store_true",dest = "operationtype",default = False)
parser.add_option("-b","--batch",action = "store_true",dest = "descenttype",default = False)
(options,args) = parser.parse_args()

#X = np.array([[1,.2,.3,.4,.5,.6,.7,.8],[0.7,0.4,0.3,0.8,0.9,0.1,0.1,0.2],[0.2,0.1,0.6,0.8,0.99,0.14,0.02,0.77],[0.1,0.99,0.1,0.1,0.1,0.7,0.5,0.3],np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8),np.random.uniform(0,1,8)])
#y = np.array([[1,.2,.3,.4,.5,.6,.7,.8],[0.7,0.4,0.3,0.8,0.9,0.1,0.1,0.2],[0.2,0.1,0.6,0.8,0.99,0.14,0.02,0.77],[0.1,0.99,0.1,0.1,0.1,0.7,0.5,0.3]])
#X = np.array([[.1,.2,.9,.6,.1],[.8,.4,.9,.6,.5],[.9,.2,.9,.3,.8],[.2,.1,.9,.6,.8],[.4,.5,.9,.0,.99],[.3,.4,.9,.6,.1],[.3,.5,.1,.1,.7],[.2,.4,.02,.98,.3],[.4,.9,.1,.8,.2]])

# Import data
X1 = get_data("rap1-lieb-positives.txt")
print("Finished parsing positive dataset")
X2temp = parse(read_negatives())
print("Finished parsing negative dataset")

def remove_duplicates(A,B):
    # Quick function to remove any sequences in the negative data that are also present in the positive data.
    cumdims = (np.maximum(A.max(),B.max())+1)**np.arange(B.shape[1])
    return A[~np.in1d(A.dot(cumdims),B.dot(cumdims))]

X2 = remove_duplicates(X2temp,X1)
rem = X2temp.shape[0] - X2.shape[0]
print("Removing negative sequences that are identical to positive dataset")
print("{} sequences removed".format(rem))
# Shuffle positive X data & create datasets for cross-validation
np.random.shuffle(X1)
X1a = X1[:28]
X1b = X1[28:56]
X1c = X1[56:83]
X1d = X1[83:110]
X1e = X1[110:137]
X1list = [X1a,X1b,X1c,X1d,X1e]

#Create answer datasets
y1a = np.ones((X1a.shape[0],1))
y1b = np.ones((X1b.shape[0],1))
y1c = np.ones((X1c.shape[0],1))
y1d = np.ones((X1d.shape[0],1))
y1e = np.ones((X1e.shape[0],1))
y1list = [y1a,y1b,y1c,y1d,y1e]

# Shuffle negative X data & create datasets for cross-validation
avgK = int(round(X2.shape[0]/5))
np.random.shuffle(X2)
X2a = X2[:avgK]
X2b = X2[avgK:2*avgK]
X2c = X2[2*avgK:3*avgK]
X2d = X2[3*avgK:4*avgK]
X2e = X2[4*avgK:5*avgK]
X2list = [X2a,X2b,X2c,X2d,X2e]


# Create answer datasets
y2a = np.zeros((X2a.shape[0],1))
y2b = np.zeros((X2b.shape[0],1))
y2c = np.zeros((X2c.shape[0],1))
y2d = np.zeros((X2d.shape[0],1))
y2e = np.zeros((X2e.shape[0],1))
y2list = [y2a,y2b,y2c,y2d,y2e]

print("Finished creating datasets")
#Initiate neural network
NN = neural_network(68,1,200)

"""
If '-t' or '--train' is not specified, the network will take the training data, now broken down
into 5 pieces, and one by one exclude a piece of training data. At the end of this code block,
the withheld section of data is used to test the network's performance.
"""
if options.operationtype == False:
    for i in range(0,5):
        print("Training dataset {}".format(i+1))
        NN.__init__(68,1,200)
        X1test = X1list[i]
        X2test = X2list[i]
        y1test = y1list[i]
        y2test = y2list[i]
        Xtrainlistpos = []
        Xtrainlistneg = []
        ytrainlistpos = []
        ytrainlistneg = []
        for j in range(0,5):
            if j != i:
                Xtrainlistpos.append(X1list[j])
                Xtrainlistneg.append(X2list[j])
                ytrainlistpos.append(y1list[j])
                ytrainlistneg.append(y2list[j])

        Xtrainpos = np.concatenate(Xtrainlistpos)
        Xtrainneg = np.concatenate(Xtrainlistneg)
        ytrainpos = np.concatenate(ytrainlistpos)
        ytrainneg = np.concatenate(ytrainlistneg)
        Xtrain = np.concatenate((Xtrainpos,Xtrainneg))
        ytrain = np.concatenate((ytrainpos,ytrainneg))


        if options.descenttype == False:
            out = NN.train_stochastic(Xtrain,ytrain,options.iter,options.alpha,options.lam,Xtrainpos.shape[0],Xtrainneg.shape[0])
        elif options.descenttype == True:
            out = NN.train(Xtrain,ytrain,options.iter,options.alpha,options.lam)
        print(out)
        Xtest = np.concatenate((X1test,X2test))
        ytest = np.concatenate((y1test,y2test))
        scores = NN.forward(Xtest)
        fpr,tpr,thresholds = metrics.roc_curve(ytest,scores)
        roc_auc = metrics.auc(fpr,tpr)
        lw = 2
        colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])

        plt.plot(fpr,tpr,color = 'black',lw = lw,label = 'ROC curve (area = {})'.format(roc_auc))
        plt.plot([0,1],[0,1],'k--',lw=lw)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristics')
        plt.legend(loc = "lower right")
        plt.show()
        np.savetxt('nptrain_out{}.txt'.format(i+1),scores)
        np.savetxt('npvalues_out{}.txt'.format(i+1),ytest)

if options.operationtype == True:
    X1 = np.concatenate(X1list)
    X2 = np.concatenate(X2list)
    y1 = np.concatenate(y1list)
    y2 = np.concatenate(y2list)
    Xtrain = np.concatenate((X1,X2))
    ytrain = np.concatenate((y1,y2))
    out = NN.train_stochastic(Xtrain,ytrain,options.iter,options.alpha,options.lam,X1.shape[0],X2.shape[0])
    # Generate an ROC curve of training data just as a sanity check to make sure things are still working.
    fpr,tpr,thresholds = metrics.roc_curve(ytrain,out)
    roc_auc = metrics.auc(fpr,tpr)
    lw = 2
    colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])

    plt.plot(fpr,tpr,color = 'black',lw = lw,label = 'ROC curve (area = {})'.format(roc_auc))
    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc = "lower right")
    plt.show()

    Xtest = get_data("rap1-lieb-test.txt")
    out = NN.forward(Xtest)
    np.savetxt('np_out.txt',out)


#print(NN.W1)
#T = trainer(NN)
#T.train(X,y)
#print(NN.yHat)
#newX = ([.1,.4,.01,.3,.4,.8,.8,.1])
#print(NN.forward(newX))

#NN.train(X,y,options.iter,options.alpha,options.lam)
