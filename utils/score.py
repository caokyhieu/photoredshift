

class Model:
    def __init__(self,params_key):
        self.params = {k:None for k in params_key}
    
    def set_params(self,params):
        """
        params: dict of params
        """
        assert list(params.keys()) == list(self.params.keys()) 
        self.params.update(params)

    def _mapping(self):
        pass

    def predict(self,x):
        """
        x: covariate arr
        """
        pass

class Scoring:

    def __init__(self,model,params):
        
        self.model = model
        self.params = params

    