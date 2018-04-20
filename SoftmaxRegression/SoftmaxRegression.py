#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model_utils import * 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


"""
@Author: Kai Bernardini

TODO: 
- Add Early Stopping
- Addm Momentum
- Add Mini Batch 
"""

class SoftmaxRegression(object):
    
    """
    Parameters
    ------------
    lmbda: float
        Regularization parameter for L2 regularization. 
        No regularization if l2=0.0.
    learning_rate : float (default: 0.01)
        Learning rate (between 1e-6 and 1.0)
    beta: float (default:.9)
        beta initialization for GD with momentum
    num_iterations : int (default: 1000)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    opt: string (default: 'GD')
        Optimization method. Supports 'GD', 'Momentum', 'MB_Momentum' for 
        Gradient descentnt, with momentum, and minibath with momentum respectively. 
    batch_size : int (default: 64)
        The number of minibatches for gradient-based optimization.
        If len(y): Stochastic Gradient Descent (SGD) online learning
    seed : int (default: 6969)
        Set random state for initializing the weights.

    Attributes
    -----------
    W : weight matrix, shape={n_features, n_classes}
      Model weights 
    b : 1d-array, shape={n_classes,}
      Bias unit after fitting.
    costs : list
        Array of costs after several training 

    Methods 
    -----------
    fit

    """

    def __init__(self, lmbda = 0, learning_rate = .1,beta=.9 ,num_iterations= 1000, opt='GD', seed=6969, batch_size=64):
        self.costs = None
        self.opt=opt
        self.lmbda = lmbda
        self.alpha = learning_rate
        # only used if momentum is used 
        self.beta=.9
        self.num_iterations = num_iterations
        self.seed = seed
        self.batch_size = batch_size
        
    def fit(self, X, y, verbose=False):
        """
        Trains the Model 

        Arguments:
        X -- Datamatrix
        y -- ground truth labels
        verbose -- boolean for perioidcally printing cost
        """
        self.p = X.shape[1]
        self.b, self.W = initialize_params(X, y)
        opt=self.opt
        one_hot =  OneHotEncoder(sparse=False)
        one_hot.fit(y)
        self.one_hot = one_hot
        Y = self.one_hot.transform(y)
        print(self.W.shape)
        if opt =="GD":
            print("Using Gradient Descent")
            self.b,self.W , self.costs = gradientDescent(
                self.b, 
                self.W,
                X,
                Y,
                self.lmbda,
                self.num_iterations,
                self.alpha,
                verbose = verbose)
        if opt == "Momentum":
            raise(NotImplementedError)
            print("Using Gradient Descent with Momentum")
            self.v = initialize_velocity(self.b,self.w)
            self.b,self.w , self.costs = momentumGradientDescent(
                self.b, 
                self.w,
                self.v,
                X,
                Y,
                self.lmbda,
                self.num_iterations,
                self.alpha,
                self.beta,
                verbose = verbose)
        if opt == "MB_Momentum":
            raise(NotImplementedError)
            print("Using MiniBatch Gradient Descent with Momentum")
            self.v = initialize_velocity(self.b,self.w)
            self.b,self.w , self.costs = miniBatchMomentumGradientDescent(
                b=self.b, 
                w= self.w,
                v= self.v,
                X=X,
                Y=Y,
                lmbda = self.lmbda,
                num_iterations= self.num_iterations,
                learning_rate= self.alpha,
                beta= self.beta,
                batch_size= self.batch_size,
                seed = self.seed,
                verbose = verbose)
            

        
    def predict_proba(self, X):
        """
        Predict the class probabilities 

        Arguments: 
        X -- Datamatrix of shape (n, p) (number of examples, predictors)

        Returns:
        probabilities matrix of shape (N, |G| ) (number of examples, number of classes)

        """
        return h(self.b, self.W, X)
    
    def predict(self, X):
        """
        Predict the class 

        Arguments: 
        X -- Datamatrix

        Returns:
        Class label
        """
        probs = self.predict_proba(X)
        return np.argmax(probs,axis=1)
    
    def score(self, X, y):

        """
        Computes Accuracy score

        Arguments: 
        X -- datamatrix
        y -- ground truth labels

        """
        preds = self.predict(X)
        return np.squeeze(np.sum(preds == y.ravel()) / len(y))
    
    def plot_learning_curve(self):
        """
        Plots and saves a png of the learning curve 

        """
        # Plot learning curve (with costs)
        plt.figure(figsize=(8,6))
        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.alpha))
        # save using time value 
        plt.savefig("Learning_Curve{}.png".format( time.time()))
        plt.show()
        
