from abc import ABC, abstractmethod
from jax import Array
from jax.random import PRNGKey


class FeynmanKac(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def M0_rvs(self, params, key: PRNGKey, inp: tuple):
        """ Implement t=0 proposal kernel for SMC """
        pass

    @abstractmethod
    def Mt_rvs(self, params, key: PRNGKey, xp, inp: tuple):
        """ Implement Markov proposal kernel for SMC """
        pass

    @abstractmethod
    def M0_logpdf(self, params, x0, inp: tuple, constant: bool):
        """
        Implement logpdf for t=0 proposal kernel

        params:    Dictionary of params required for Markov kernel evaluation.
        x0:        t=0 state vector
        inp:       Any external inputs required for logpdf evaluation e.g. ys[0]
        constant:  Whether the calculate the normalising constant for the logpdf
        """
        pass

    @abstractmethod
    def Mt_logpdf(self, params, xp, x, inp: tuple, constant: bool): 
        """
        Implement logpdf for Markov proposal kernel

        params:    Dictionary of params required for Markov kernel evaluation.
        xp:        Previous state vector
        x:         Current state vector
        inp:       Any external inputs required for logpdf evaluation e.g. ys[t]
        constant:  Whether the calculate the normalising constant for the logpdf
        """

    @abstractmethod
    def G0_logpdf(self, params, x0, inp: tuple):
        pass 

    @abstractmethod
    def Gt_logpdf(self, params, x, inp: tuple):
        """ Implement logpdf for potential function """
        pass

    def Gamma_0(self, params, x0, inp, constant: bool):
        return self.G0_logpdf(params, x0, inp) + self.M0_logpdf(params, x0, inp, constant=constant)

    def Gamma_t(self, params, xp, x, inp, constant: bool):
        return self.Gt_logpdf(params, x, inp) + self.Mt_logpdf(params, xp, x, inp, constant=constant)

    @abstractmethod
    def init(self, key: PRNGKey, params: dict, data: tuple[Array], **kwargs):
        """ 
        Write a function to get first state for SMC. 
        Usually unconditional run 
        """
        pass

    @abstractmethod
    def get_kernel(self, params, state, data, conditional: bool, **kwargs):
        """ Write a kernel constructor using defined FK methods """
        pass


class SMC(ABC):

    def __init__(self, fk: FeynmanKac, conditional: bool, kwargs: dict):
        self.fk = fk
        self.conditional = conditional
        self.kwargs = kwargs

    def init(self, key: PRNGKey, params: dict, data: tuple[Array]):
        return self.fk.init(key, params, data, **self.kwargs)

    def sample(
            self, 
            key: PRNGKey, 
            params: dict,
            state: tuple[Array],
            data: tuple[Array],
        ):

        # construct new kernel with params
        kernel = self.fk.get_kernel(
            params,
            state, 
            data, 
            conditional=self.conditional, 
            **self.kwargs
        )

        # sample new smoothing path
        xs, Bs, log_ws = kernel(key)

        # aux
        prev_Bs = state[-1]
        replaced = Bs != prev_Bs
        return (xs, Bs), {"replaced": replaced, "log_ws": log_ws}
         

        