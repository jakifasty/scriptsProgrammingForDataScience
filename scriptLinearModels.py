#gradient descent method algotithm

import numpy as np
import pandas as pd

def gradientDescent():
    #Input: none
    #Output: none
    
    #returns the gradient descent method
    #for the function f(x) = (x-2)^2
    #with the learning rate alpha = 0.1
    #and the initial value x = 0

    x = 0 #initial value
    alpha = 0.1
    nu = 0.0 #tolerance
    while np.abs(np.gradient(x)) < nu:
        [w0, w1] = - np.gradient(x)
        x = x - alpha*2*(x-2)
        print("x: {}".format(x))
    return (np.gradient)

def LinearDescentForLinearRegression(m, w0, wi, t):
    return (None)


