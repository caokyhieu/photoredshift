


def prior_checking(model,prior,data):
    """
    data: A dictionary data, which require for your model(both covariate and rensponse)
    prior: a prior you want to set for model
    both need in from of dictionary
    """
    assert isinstance(prior,dict),"prior must in dict form"
    assert isinstance(data,dict),"data must in dict form"
    assert set(data.keys()).intersection(set(model.__code__.co_varnames)) == set(data.keys()),\
        'data need have key as the argument name in model'
    assert None in data.values(),"you need to set None to obs sample"
    data.update({'prior':prior})
    return model(**data)



