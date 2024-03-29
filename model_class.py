import numpy as np
from hmmlearn import hmm

# https://github.com/TOMILO87/time_series_simulation/blob/master/ARIMA
def ARIMA(phi = np.array([0]), theta = np.array([0]), d = 0, t = 0, mu = 0, sigma = 1, n = 20, burn = 10):
    """ Simulate data from ARMA model (eq. 1.2.4):

    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p} + a_t + theta_1*a_{t-1} + ... + theta_q*a_{t-q}

    with d unit roots for ARIMA model.

    Arguments:
    phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
    theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
    d -- number of unit roots for non-stationary time series
    t -- value deterministic linear trend
    mu -- mean value for normal distribution error term
    sigma -- standard deviation for normal distribution error term
    n -- length time series
    burn -- number of discarded values because series beginns without lagged terms

    Return:
    x -- simulated ARMA process of shape (n, 1)

    Reference:
    Time Series Analysis by Box et al.
    """

    # add "theta_0" = 1 to theta
    theta = np.append(1, theta)
    
    # set max lag length AR model
    p = phi.shape[0]

    # set max lag length MA model
    q = theta.shape[0]

    # simulate n + q error terms
    a = np.random.normal(mu, sigma, (n + max(p, q) + burn, 1))
    
    # create array for returned values
    x = np.zeros((n + max(p, q) + burn, 1))

    # initialize first time series value
    x[0] = a[0]

    for i in range(1, x.shape[0]):
        AR = np.dot(phi[0 : min(i, p)], np.flip(x[i - min(i, p) : i], 0))
        MA = np.dot(theta[0 : min(i + 1, q)], np.flip(a[i - min(i, q - 1) : i + 1], 0))
        x[i] = AR + MA + t

    # add unit roots
    if d != 0:
        ARMA = x[-n: ]
        m = ARMA.shape[0]
        z = np.zeros((m + 1, 1)) # create temp array

        for i in range(d):
            for j in range(m):
                z[j + 1] = ARMA[j] + z[j] 
            ARMA = z[1: ]
        x[-n: ] = z[1: ]
        
    return x[-n: ]

class ARIMA_model:
    """Autoregressive integrated moving average model.

    https://github.com/TOMILO87/time_series_simulation/blob/master/ARIMA
    """
    def __init__(self, phi, theta, d = 0, t = 0, mu = 0, sigma = 1, burn = 10):
        self.phi = phi
        self.theta = theta
        self.d = d
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.burn = burn

    def sample(self, n):
        return ARIMA(self.phi, self.theta, self.d, self.t, self.mu, self.sigma, n, self.burn)

    @property
    def p(self):
        return len(self.phi)

    @property
    def q(self):
        return len(self.theta)

class ARIMA_HMM_model:
    """Mixture of tau*ARIMA and (1-tau)*HMM model."""
    def __init__(self, phi, theta, d = 0, t = 0, mu = 0, sigma = 1, burn = 10, tau: float = 0.3, HMM_instance: hmm.BaseHMM = None):
        if HMM_instance is None:
            raise NotImplementedError(f"must pass HMM instance")
        
        self.phi = phi
        self.theta = theta
        self.d = d
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.burn = burn
        self.HMM_instance = HMM_instance
        self.tau = tau

    def sample(self, n):
        ARIMA_sample = ARIMA(self.phi, self.theta, self.d, self.t, self.mu, self.sigma, n, self.burn)
        HMM_sample, _ = self.HMM_instance.sample(n)
        return self.tau*ARIMA_sample + (1-self.tau)*HMM_sample


    @property
    def p(self):
        return len(self.phi)

    @property
    def q(self):
        return len(self.theta)