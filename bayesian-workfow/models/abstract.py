from abc import ABC, abstractmethod

class Distribution(ABC):

    @abstractmethod
    def sample(self,*args,**kwargs):
        pass 

    @abstractmethod
    def set_parameters(self,*args,**kwargs):
        pass 

    @abstractmethod
    def get_parameters(self,*args,**kwargs):
        pass 

    @abstractmethod
    def _cdf(self):
        pass 

    @property
    def cdf(self):
        return self._cdf 

    @abstractmethod
    def _pdf(self):
        pass 

    @property
    def pdf(self):
        return self._pdf 
    

class AbstractModel(ABC):

    @classmethod
    def set_parameters(cls,**kwargs):
        "params input is distribution with prior"
        cls.__dict__.update({'params':kwargs})
        pass
    
    @abstractmethod
    def likelihood(self,**kwargs):
        """
        Function of parameters
        """
        # assert kwargs.keys() == self.params.keys(), f"Params for likelihood {kwargs.keys()} != params model {self.params.keys()}"
        pass
    
    @abstractmethod
    def posterior(self):
        pass 
    
    @abstractmethod
    def fit(self,data):
        pass


