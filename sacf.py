# https://github.com/TOMILO87/time_series_simulation/blob/master/sacf
import numpy as np

def sacf(X, k = 10, print_sacf = False):
    """Calculat sample autocorrelation function (acf) at different lags

    Arguments:
    X -- time series of shape (length time series, number of time series)
    k -- lag length, an integer
    print_sacf -- prints sample auto correlation function if true

    Return:
    sacf -- sample auto correlation functions of shape (lag length, number of time series)
    """

    # length time series is n and numer of times series is m
    n, m = X.shape

    # warning if lag length is less than length time series
    if k > n:
        print("\nWarning: length of time series", n, "is less than lag length", k,)
        if k == 10:
            print("Defualt lag length is", k, "\n")

    # mean values
    mu = np.mean(X, axis = 0)

    # variances corresponding to zero lag
    var = np.var(X, axis = 0)

    # autocovariance at different lags
    cov = np.zeros((k, m))
    
    for j in range(0, m):
        for i in range(1, k + 1):
            cov[i - 1, j] = np.dot(np.transpose(X[:-i, j] - mu[j]), X[i:, j] - mu[j]) / n

    # sample autocorrelation at different lags
    sacf = cov / var

    if print_sacf:
        print(sacf)

    return sacf