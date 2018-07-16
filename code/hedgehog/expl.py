"""
Code for Project 1
"""

import numpy as np

#
# Convergence Testing for Root-Finding
#

def x_ratio_err(x1, x2, tol):
	'''
	Ratio of last two root estimates
	'''
	if x1 != 0:
		err = (x2 - x1) / x1
	else:
		err = tol + 1
	return np.abs(err)

def has_converged(x1, x2, tol, test="x_ratio", isVerbose=False):
	'''
	Checks if the sequence has converged
		x1   - Old estimate
		x2   - New estimate
		tol  - accepted tolerance (error) value
		test - what type of error test should be performed
			Defaults to 'x_ratio'
	'''

	if test == "x_ratio":
		err = x_ratio_err(x1, x2, tol)

	if isVerbose:
		print("Error: {:.8f}".format(err))
	if err <= tol:
		return True
	return False

## Newton-Raphson

def newton_raphson(f, fp, x0, max_iter=None, tol=1e-4, test="x_ratio", isVerbose=False):
	'''
	Newton-Raphson method for root finding
		f         - function whose root is to be found
		fp        - first derivative of function f
		x0        - initial root guess
		max_iter  - maximum number of iterations before stopping
		 	Defaults to infinte (which is bad...)
		tol       - tolerance value for convergence
		test      - type of test used to check for convergence.
			Defaults to 'x_ratio'
		isVerbose - Prints results from each iteration
	'''

	if max_iter is None:
		max_iter = float("inf")

	# loop until convergence or max_iter
	i = 1
	while i <= max_iter:

		# update best guess
		if fp(x0) != 0:
			x1 = x0 - f(x0) / fp(x0)
		else:
			x1 = float("-inf")
		if isVerbose:
			print("Iter: {:3}\t Best Guess: {:.7f}".format(i, x1))

		# check for convergence or divergence
		if has_converged(x0, x1, tol=tol, test=test):
			if isVerbose:
				print("Converged")
				has_converged(x0, x1, tol=tol, test=test, isVerbose=isVerbose)
			return x1
		if x1 == float("inf") or x1 == float("-inf") or x1 == float("nan"):
			if isVerbose:
				print("Divergent")
			return x1
		
		# update values
		temp = x0
		x0 = x1
		i += 1

	has_converged(temp, x1, tol=tol, test=test, isVerbose=isVerbose)
	return x0

## Simpson's Rule

def simpson_1_3(f, x0, xn, h):
	''' Use Simpson's 1/3 rule to approximate the integral of f '''

	X = np.arange(x0, xn, step=h, dtype=float)  # create list of points
	Y = [f(x) for x in X]                       # create list of function values
	Y.append(f(xn))                             # end point is missed by arange
	n = len(Y)
	ans = 0.0
	i = 0
	while i < n-2:
		ans += Y[i] + 4*Y[i+1] + Y[i+2]
		i += 2
	ans *= h / 3

	return ans


## RK-4

def runge_kutta_4(system, u0, h, t0, tf, isVerbose=True, retAll=False):
	'''
	Runge-Kutta Order 4
	system and u0 are numpy arrays of same size
	h is step size
	t0 is initial time
	tf is final time
	'''

	assert len(system) == len(u0)               # error check

	n = len(system)                             # size of the system
	system = list(system.reshape(n,))           # convert into list for list comprehensions
	ti = t0                                     # initial time
	ui = u0                                     # initial conditions
	N = int((tf - ti) / h)                      # number of steps
	U = [u0]
	T = [t0]

	for i in range(N):

		# calculate the different k's
		k1 = h * np.array([f(ti, ui) for f in system]).reshape(n, 1)
		k2 = h * np.array([f(ti + h/2, ui + k1/2) for f in system]).reshape(n, 1)
		k3 = h * np.array([f(ti + h/2, ui + k2/2) for f in system]).reshape(n, 1)
		k4 = h * np.array([f(ti + h, ui + k3) for f in system]).reshape(n, 1)

		# print function values at different times
		if isVerbose:
			output = "t = {:.7f}\t( ".format(ti)
			for i in range(n-1):
				output += "{:.7f}, ".format(ui[i][0])
			output += "{:.7f} )".format(ui[n-1][0])
			print(output)

		# update step
		ui = ui + (k1 + 2*k2 + 2*k3 + k4) / 6
		ti += h
		U.append(ui)
		T.append(ti)

	# print final value
	if isVerbose:
			output = "t = {:.7f}\t( ".format(ti)
			for i in range(n-1):
				output += "{:.7f}, ".format(ui[i][0])
			output += "{:.7f} )".format(ui[n-1][0])
			print(output)

	if retAll:
		return ui, U, T
	else:
		return ui
