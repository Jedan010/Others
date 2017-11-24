# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:00:04 2017

@author: J
"""

import numpy as np

##sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

##relu
def relu(z): 
    return np.maximum(0, z)

##relu_backward
def relu_backward(z):
    dz = np.array(z, copy=True)
    dz[dz>0] = 1
    dz[dz<=0] = 0
    
    return dz

def sigmoid_backward(z):
    return sigmoid(z) * (1- sigmoid(z))



##initialize_parameters
def initialize_parameters(layer_dims):
    #layer_dims -- python array (list) containing the dimensions
    #of each layer in our network
    
    parameters = {}
    L = len(layer_dims)    # number of layers in the network

    for l in range(1, L):        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

##forward propagation
def forward_propagation(X, parameters, activations):
    A_pre = X
    L = len(parameters) // 2
    caches = []
    for i in range(L):
        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        activation = activations[i]
        
        Z = np.dot(W, A_pre) + b
        if activation == 'sigmoid':
            A = sigmoid(Z)
        elif activation == 'relu':
            A = relu(Z)
        
        A_pre = A
        
        caches.append({'A': A, 'Z': Z})        
    
    return caches

def cost_function(yhat, y):
    m = y.shape[1]
    
    cost = -1/ m * np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))
    cost = np.squeeze(cost)
    
    return cost

def backward_gradient(dA, A_pre, Z, W, activation):
    
    m = Z.shape[1]
    
    if activation == 'sigmoid':
        dZ = dA * sigmoid_backward(Z)
    elif activation == 'relu':
        dZ = dA * relu_backward(Z)
       
    dW = 1 / m *np.dot(dZ, A_pre.T)
    db = 1 / m *np.sum(dZ, axis=1, keepdims=True)
    dA_pre = np.dot(W.T, dZ)
    
    return dA_pre, dW, db


    
def backward_propagation(y, parameters, caches, activations):
    L = len(parameters) // 2
    
    grads = {}
    
    ## initialize dA_pre
    yhat = caches[L]['A']
    dA = -(np.divide(y, yhat) + np.divide(1-y, 1-yhat))
    
    dA_pre, dW, db = backward_gradient(dA, caches[L-2]['A'], caches[L-1]['Z'],
                                   parameters['W'+str(L)], activations[L-1])
    
    grads['dW'+str(L)], grads['db'+str(L)] = dW, db
   
    
    for i in range(1, L-1):
        dA = dA_pre
        dA_pre, dW, db = backward_gradient(dA, caches[i-1]['A'], caches[i]['Z'],
                                   parameters['W'+str(i+1)], activations[i])
        grads['dW'+str(i+1)], grads['db'+str(i+1)] = dW, db
        
        
    return grads

def updata_parameter(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for i in range(1,L+1):      
        parameters['W'+str(i)] -= grads['dW'+str(i)] * learning_rate
        parameters['b'+str(i)] -= grads['db'+str(i)] * learning_rate
            
    return parameters       


## train the models 
def train(X, y, layer_dims, n_iter, learning_rate, activations):
    ## intitialize parameters
    parameters = initialize_parameters(layer_dims)
    
    ## L layres
    L = len(layer_dims)
    
    costs = [] ##record cost
    ## train model
    for i in range(n_iter):
    
        ## forward propagation
        caches = forward_propagation(X, parameters, activations)
        
        ## cost 
        cost = cost_function(caches[L-1]['A'], y) 
        costs.append(cost)
        
        ## backward propagation
        grads = backward_propagation(y, parameters, caches, activations)
        
        ## update parameters
        parameters = updata_parameter(parameters, grads, learning_rate)
        
def predict(X, parameters, activations):
    caches = forward_propagation(X, parameters, activations)
    yhat = caches[-1]['A']
    
    return yhat

def accuracy(yhat, y):
    return 1 - np.mean(np.abs(yhat - y))