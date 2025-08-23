#!/usr/bin/env python3
# Functions for random matrix theory: compute Marčenko–Pastur bounds for sample covariance eigenvalues
def mp_bounds(N, T, sigma2=1.0):
 if N > T:
    raise ValueError("N must be less than or equal to T")
 #return the bounds for the eigenvalues
 c = N / T
 lambda_min = sigma2 * (1 - c)**2
 lambda_max = sigma2 * (1 + c)**2
 return (lambda_min, lambda_max)
#example usage
fun = mp_bounds(10, 100)
print("the bounds are: ", fun)