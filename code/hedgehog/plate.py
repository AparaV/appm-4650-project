"""
Code for Project 2
"""

import numpy as np

from hedgehog.expl import *

## Interpolating Polynomials

def lagrange_poly(X, Y, a, isVerbose=False):
	''' Construct the lagrange approximation at a using data points '''

	assert len(X) == len(Y)
	
	ans = 0.0
	for idx, x in enumerate(X):
		prod = Y[idx]
		for x_i in X:
			if x_i != x:
				prod *= (a - x_i) / (x - x_i)
		ans += prod

	if isVerbose:
		print("f({}) = {:.6f} using Lagrange polynomial approximation of degree {}".
			format(str(a), ans, len(X)-1))

	return ans