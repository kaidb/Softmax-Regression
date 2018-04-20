#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm



def initialize_params(X, y):
    """
    Initialize weights and bias for softmax regression
    
    Arguments: 
    X -- datamatrix composed of n examples and p predictors of shaoe (number of examples (n) , number of predictors (p)) 
    y --  ground truth labels (array) 
    
    Returns:
    b -- bias initialized to zero
    W -- weight matrix of shape (p, |G|) where |G| is the number of classes
    """
    
    n_classes = len(np.unique(y))
    W = np.random.randn(X.shape[1], n_classes) * .1
    b = np.zeros((n_classes))
    return b,W


def softmax(x):
    """
    Calculates the softmax for each row of the input x.

    Argument:
    x -- A numpy matrix of shape (n,m)
    
    Returns:
    s -- A numpy matrix equal to the softmax of x,
    """
    e_x = np.exp(x - np.max(x, keepdims=True,  axis=1))
    s_ex = np.sum(e_x, keepdims=True, axis=1) 
    sm =  e_x / s_ex.reshape((len(s_ex), 1))
    return sm


def h(b, W ,X):
    """
    This function implments the softmax regression hypothesis function

    Argument:
    b -- bias
    W -- predictive weight matrix 
    X -- data matrix of size (numbers_examples, number_predictors)

    Returns:
    softmax(XW + b)
    """
    return softmax( (X @ W) + b)



def computeCostGrad(b, W, X, Y, lmbda = 0): 
    """
    Computes Cross Entropy Loss function with l2 regularization 

    Arguments:
    b -- bias vector
    w -- predictive weight matrix 
    X -- data matrix of size (numbers_examples, number_predictors)
    Y -- Ground truth matrix labels of size (number_examples, |G|)
    lmbda -- regularization hyperparameter
    
    Return:
    cost -- negative log-likelihood cost for logistic regression
    grad -- gradient 
    """

    m = Y.shape[0]
    # contribution to cost from regularzing lambda
    reg = (1/2)* lmbda *  np.sum(  np.multiply(W , W).ravel() )
    # prediction probabilities
    prob = h(b, W ,X)
    loss = (-1 / m) * np.multiply(Y, np.log(prob) ) 
    loss += reg
    cost = np.sum(loss.ravel())
    #print(type(loss))
    #cost = np.sum( loss.ravel() )
    #print(loss.shape)
    dW = (1/m ) *  X.T @ (prob -Y) + (lmbda * W) 
    db =  (1/m ) *  np.sum(prob-Y, axis=0, keepdims=True)
    grads = {'dW': dW, 'db': db}
    return  grads,cost



# Python implementation of gradient descent with early stoping 
def gradientDescent(b, W,  X, Y,lmbda, num_iterations, learning_rate, verbose = False, stopping_tolerance = 1e-8):
    """
    This function calculates w,b  by running a gradient descent for num_iterations. 
    
    Arguments:
    b -- bias, 
    W -- predictors
    X -- datamatrix
    Y -- ground truth labels matrix
    lmbda -- regularization hyperparameter
    num_iterations -- number of times to run GD 
    learning_rate -- values for alpha. 
    verbose --Log the loss every 100 steps
    
    Returns:
    b_optimal -- optimized bias 
    W_optimal -- optimized predictor
    costs -- Array of costs
    
    TODO: add early stoping, optimization options, regularization 
    """
    
    costs = []
    tol=0
    for i in tqdm(range(num_iterations)):
        # Compute Gradient and cost
        grads, cost= computeCostGrad(b, W, X, Y, lmbda)
        # Retrieve derivatives from grads
        dW = grads["dW"]
        db = grads["db"]
        
        # Gradient update
        W_old = W
        b_old = b
        
        W = W - learning_rate * (dW)
        b = b - learning_rate * (db)
        #print(b.shape)
        #assert(b.shape == ())
        if 1!=1:
            pass
        else:
            tol=0
        # Log cost (every 100 iterations )
        if i % 100 == 0:
            costs.append(cost)
            if verbose:
                #print("Loss shape ", cost.shape)
                print ("Loss Value after iteration {}: {}, with shape {}".format(i, cost, cost.shape))
    
    b_optimal, W_optimal = b, W
    return b_optimal, W_optimal, costs
