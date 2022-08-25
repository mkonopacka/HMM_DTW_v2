from ARIMA import *
from hmmlearn import hmm

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