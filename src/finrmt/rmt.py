#!/usr/bin/env python3
# Functions for random matrix theory: compute Marčenko–Pastur bounds for sample covariance eigenvalues
import numpy as np
def mp_bounds(N, T, sigma2):
 if N > T:
    raise ValueError("N must be less than or equal to T")
 #return the bounds for the eigenvalues
 c = N / T
 q = np.sqrt(c)
 lambda_min = sigma2 * (1 - q)**2
 lambda_max = sigma2 * (1 + q)**2
 return (lambda_min, lambda_max)