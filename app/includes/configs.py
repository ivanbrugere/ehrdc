import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_base_config(model_fn=None, model_params={}):
    config = {}
    if model_fn is None:
        config["model_fn"] = GradientBoostingClassifier
    else:
        config["model_fn"] = model_fn
    config["model_params"] = model_params
    config["model"] = config["model_fn"](**model_params)
    config["train path"] = "../train/"
    config["test path"] = "../infer/"
    config["model path"] = "../model/"
    config["output path"] = "../output/"
    config["train"] = True
    return config

def get_rf_baseline_config(model_params={"max_depth": 100, "n_estimators":200, "n_jobs":-1}):
    config = get_base_config(RandomForestClassifier, model_params)
    config["model name"] = "static simple"
    return config

def get_naivebayes_baseline_config():
    config = get_base_config(BernoulliNB)
    config["model name"] = "static simple"
    return config

def get_baseline_cv_configs():
    configs = dict()
    #configs["gb"] = get_base_config()
    #configs["knn-25"] = get_base_config(model_fn=KNeighborsClassifier,
    #                                                  model_params={"n_neighbors": 25})
    #configs["knn-50"] = get_base_config(model_fn=KNeighborsClassifier,
    #                                                  model_params={"n_neighbors": 50})
    #configs["knn-100"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 100})
    #configs["knn-200"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 200})
    configs["knn-300"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 300})
    configs["knn-500"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 500})
    configs["knn-1000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 1000})
    configs["knn-2000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 2000})
    #configs["nb"] = get_base_config(model_fn=BernoulliNB)
    configs["nb-Compliment"] = get_base_config(model_fn=ComplementNB)
    #configs["nb-Multi"] = get_base_config(model_fn=MultinomialNB)
    #configs["random stratified"] = get_base_config(model_fn=DummyClassifier,
    #                                                             model_params={"strategy": "stratified"})
    #configs["random uniform"] = get_base_config(model_fn=DummyClassifier,
    #                                                          model_params={"strategy": "uniform"})
    configs["rf"] = get_rf_baseline_config()
    return configs