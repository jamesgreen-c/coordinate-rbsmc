from abc import ABC, abstractmethod
from jax.random import PRNGKey


class Prior(ABC):

    # TODO: hyperparams?
    params: dict

    @abstractmethod
    def init(self, key: PRNGKey, params: dict, data, config):
        """ 
        Implement the init function
            - set fixed parameters 
            - return parameters to optimise 
        """
        pass

    @abstractmethod
    def log_p0(self, params: dict, x0, constant: bool = True):
        """ Implement the t=0 logpdf for states """
        pass

    @abstractmethod
    def log_pt(self, params: dict, xp, x, dt, constant: bool = True):
        """ Implement the prior transition logpdf for states """
        pass

    @abstractmethod
    def log_ht(self, params, x, data):
        """ Implement the prior emission density """
        pass

    @abstractmethod
    def theta_logpdf(self, params: dict):
        """ Implement the prior parameter densities """
        pass
