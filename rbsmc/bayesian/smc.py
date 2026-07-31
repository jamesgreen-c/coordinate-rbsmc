from abc import ABC, abstractmethod
from jax import Array
from jax.random import PRNGKey


class FeynmanKac(ABC):

    def __init__(self, params):
        self.params = params

    def update(self, params):
        # TODO: for jax.jit does this need to return a new FK instance?
        self.params = {**self.params, **params}    

    @abstractmethod
    def M0_rvs(self, key: PRNGKey, inps: tuple):
        """ Implement t=0 proposal kernel for SMC """
        pass

    @abstractmethod
    def Mt_rvs(self, key: PRNGKey, xp, inps: tuple):
        """ Implement Markov proposal kernel for SMC """
        pass

    @abstractmethod
    def M0_logpdf(self, x0, inps: tuple):
        """ Implement logpdf for t=0 proposal kernel """
        pass

    @abstractmethod
    def Mt_logdf(self, xp, x, inps: tuple): 
        """ Implement logpdf for Markov proposal kernel """

    @abstractmethod
    def G0_logpdf(self, x0, inps: tuple):
        pass 

    @abstractmethod
    def Gt_logpdf(self, x, inps: tuple):
        """ Implement logpdf for potential function """
        pass

    def Gamma_0(self, x0, inps):
        return self.G0_logpdf(x0, inps) + self.M0_logpdf(x0, inps)

    def Gamma_t(self, xp, x, inps):
        return self.Gt_logpdf(x, inps) + self.Mt_logdf(xp, x, inps)

    @abstractmethod
    def init(self, key: PRNGKey, data: tuple[Array], **kwargs):
        """ 
        Write a function to get first state for SMC. 
        Usually unconditional run 
        """
        pass

    @abstractmethod
    def get_kernel(self, state, data, conditional: bool, **kwargs):
        """ Write a kernel constructor using defined FK methods """
        pass


class SMC(ABC):

    def __init__(self, fk: FeynmanKac, conditional: bool, kwargs: dict):
        self.fk = fk
        self.conditional = conditional
        self.kwargs = kwargs

    def init(self, key: PRNGKey, params: dict, data: tuple[Array]):
        self.fk.update(params)
        return self.fk.init(key, data, **self.kwargs)

    def sample(
            self, 
            key: PRNGKey, 
            params: dict,
            state: tuple[Array],
            data: tuple[Array],
        ):
        # update model parameters
        self.fk.update(params)

        # construct new kernel with params
        kernel = self.fk.get_kernel(
            state, 
            data, 
            conditional=self.conditional, 
            **self.kwargs
        )

        # sample new smoothing path
        xs, Bs = kernel(key)

        # aux
        prev_Bs = state[-1]
        replaced = Bs != prev_Bs
        return (xs, Bs), {"replaced": replaced}
         

        