import numpy as np
from scipy import optimize
import random
import progressbar


"""
Neural network class. I used a class for easy storage and retrieval of variables like
the weight/bias matrices/vectors.
"""
class neural_network(object):

    def __init__(self,inputlayersize,outputlayersize,hiddenlayersize):
        # Initiation: defines layer sizes and initiates the weights as random numbers to avoid symmetry.
        self.inputlayersize = inputlayersize
        self.outputlayersize = outputlayersize
        self.hiddenlayersize = hiddenlayersize

        self.W1 = np.random.randn(self.inputlayersize,self.hiddenlayersize)
        self.W2 = np.random.randn(self.hiddenlayersize,self.outputlayersize)
        self.bias1 = np.random.randn(1,self.hiddenlayersize)
        self.bias2 = np.random.randn(1,self.outputlayersize)

    def forward(self,X):
        #Forward propogation: input layer is multiplied by the first weight matrix and added to the bias vector to get matrix a2.
        # a2 is run through sigmoid function to get z2.
        # z2 is multiplied by second weight matrix and added to the second bias vector to get matrix z3.
        # Hypothesis yHat is found by multiplying matrix z3 by the sigmoid function.
        self.z2 = np.dot(X,self.W1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2) + self.bias2
        yHat = self.sigmoid(self.z3)

        return yHat


    def forward_stochastic(self,X,k):
        # Same as above, but this code runs the forward function for only one example of data.
        self.z2 = np.dot(X[k],self.W1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2) + self.bias2
        yHat = self.sigmoid(self.z3)

        return yHat

    def sigmoid(self,z):
        #Sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        # Derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costfunction(self,X,y,lam):
        #Compute cost for given X,y using current weights
        m = len(X[0])
        self.yHat = self.forward(X)

        J = (1/m) * np.sum((y - self.yHat)**2) + ((lam/2) * (np.sum(self.W2**2) + np.sum(self.W1**2)))#

        return J
    def costfunction_stochastic(self,X,y,k,lam):
        #Same as above costfunction, but this function computes the cost function for a single example of the data.
        m = len(X[0])
        self.yHat_s = self.forward_stochastic(X,k)
        J = np.sum((y[k] - self.yHat_s)**2) + ((lam/2) * (np.sum(self.W2**2) + np.sum(self.W1**2)))

        return J

    def costfunctionprime(self,X,y):
        # Computes the derivatives with respect to W1, W2, b1, and b2.
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)
        dJdb1 = delta2
        dJdb2 = delta3

        return dJdW1,dJdW2,dJdb1,dJdb2

    def costfunctionprime_stochastic(self,X,y,k):
        # Computes the derivatives with respect to W1, W2, b1 and b2, for a single example of data.
        self.yHat = self.forward_stochastic(X,k)
        delta3 = np.multiply(-(y[k]-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X[k].T.reshape(self.inputlayersize,1),delta2)
        dJdb1 = delta2
        dJdb2 = delta3

        return dJdW1,dJdW2,dJdb1,dJdb2


    def train(self,X,y,iterations,alpha,lam):
        # Batch gradient descent training function.
        self.X = X
        self.y = y
        m = X.shape[0]
        lim = 9e-12
        bar = progressbar.ProgressBar(maxval = iterations, widgets = [progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
        bar.start()
        count = 0
        for i in range(0,iterations):
            count +=1
            #print("iteration",i)
            bar.update(i+1)
            # At the start of each iteration, set the change in parameters to 0 matrices/vectors.
            dW1 = np.zeros((self.inputlayersize,self.hiddenlayersize))
            dW2 = np.zeros((self.hiddenlayersize,self.outputlayersize))
            db1 = np.zeros((1,self.hiddenlayersize))
            db2 = np.zeros((1,self.outputlayersize))


            #print("starting partial derivative calculation")
            # Use the cost function derivative to get the partial derivatives for all parameters.
            dJdW1,dJdW2,dJdb1,dJdb2 = self.costfunctionprime(X,y)
            #print("finished computing partial derivatives")
            # Add the partial derivatives to "d[parameter]"
            dW1 += dJdW1
            dW2 += dJdW2
            for j in range(0,(m-1)):
                # For the bias vectors, we need to do this iteratively
                db1 += dJdb1[j]
                db2 += dJdb2[j]
            #print("updating parameters")
            # Once we have the partial derivatives summed over all data examples, use the below functions to alter the parameters.
            self.W1 = self.W1 - alpha*(((1/m) * dW1) + lam * self.W1)
            self.W2 = self.W2 - alpha*(((1/m) * dW2) + lam * self.W2)
            self.bias1 = self.bias1 - alpha*((1/m) * db1)
            self.bias2 = self.bias2 - alpha*((1/m) * db2)
            #print("finished updating parameters")

            #if count == 10 and i >= 20:
                #check = self.costfunction(X,y,lam)
                #count = 0
                #if check <= 5:
                    #print("-------------------------------------converged-------------------------------------")
                    #print("score")
                    #print(check)
                    #return self.forward(X)
        bar.finish()
        print("score")
        print(self.costfunction(X,y,lam))
        #print("weights 1",self.W1)
        return self.forward(X)
    def train_stochastic(self,X,y,iterations,alpha,lam,len1,len2):
        self.X = X
        self.y = y
        m = X.shape[0]
        lim = 9e-12
        pos = 0
        neg = 0
        # Progres bar! Because I was sick of fast-scrolling iteration numbers.
        bar = progressbar.ProgressBar(maxval = iterations, widgets = [progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
        bar.start()
        for i in range(0,iterations):
            bar.update(i+1)
            rn = random.uniform(0,1) # This is the random number that determines whether positive or negative data will be used this iteration.
            if rn > 0.3:
                rn2 = random.randint(0,len1-1) # If it's greater than 0.3, use positive data.
            else:
                rn2 = random.randint(len1,len1 + len2-1) # If less, use negative.
            # Positive vs negative data is determined by the index of the numpy array. The way the data is set up, positive data comes before neagtive.

            cf = self.costfunction_stochastic(X,y,rn2,lam)
            alf = (2*alpha * cf) + alpha #Use cost function value to alter alpha.
            # Set update matrices to zero each iteration
            dW1 = np.zeros((self.inputlayersize,self.hiddenlayersize))
            dW2 = np.zeros((self.hiddenlayersize,self.outputlayersize))
            db1 = np.zeros((1,self.hiddenlayersize))
            db2 = np.zeros((1,self.outputlayersize))

            # Find partial derivatives
            dJdW1,dJdW2,dJdb1,dJdb2 = self.costfunctionprime_stochastic(X,y,rn2)
            # Update "change" matrices
            dW1 += dJdW1
            dW2 += dJdW2
            db1 += dJdb1
            db2 += dJdb2

            # And finally, update the weight and bias matrices. Since they are class objects, these will be stored outside of this function for use with test data.
            self.W1 = self.W1 - alf*(((1/m) * dW1) + lam * self.W1)
            self.W2 = self.W2 - alf*(((1/m) * dW2) + lam * self.W2)
            self.bias1 = self.bias1 - alf*((1/m) * db1)
            self.bias2 = self.bias2 - alf*((1/m) * db2)


        bar.finish()
        print("score")
        print(self.costfunction(X,y,lam))
        #print("fraction positive:",pos/(pos+neg))
        return self.forward(X)
