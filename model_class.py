from ARIMA import *

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