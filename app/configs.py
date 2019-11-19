import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import os
import numpy as np
from autosklearn import classification as ask


os.environ["OMP_NUM_THREADS"] = "8"
def get_base_config(model_fn=None, model_params={}, name=None):
    config = {}
    if name is None:
        config["model name"] = "static uid model selection"
    else:
        config["model name"] = name

    if model_fn is None:
        config["model_fn"] = GradientBoostingClassifier
    else:
        config["model_fn"] = model_fn
    config["model_params"] = model_params
    config["model"] = config["model_fn"](**model_params)
    config["train path"] = "../train_new/"
    config["test path"] = "../infer_new/"
    config["model path"] = "../model/"
    config["output path"] = "../output/"
    config["scratch path"] = "../scratch/"
    config["train"] = True
    return config

def get_rf_baseline_config(model_params={"max_depth": 100, "n_estimators":200,"n_jobs":-1}, name=None):
    config = get_base_config(RandomForestClassifier, model_params, name=name)
    return config

def get_naivebayes_baseline_config(model_params={}, name=None):
    config = get_base_config(BernoulliNB, model_params, name=name)
    return config

def get_xgboost_baseline_config(model_params={"max_depth":10, "n_jobs:":-1, "n_estimators":100}, name=None):
    return get_base_config(xgb.sklearn.XGBClassifier, model_params, name=name)



def get_baseline_cv_configs():
    configs = dict()
    configs["auto"] = get_base_config(model_fn=ask.AutoSklearnClassifier, model_params={"time_left_for_this_task":3000, "per_run_time_limit":500,
                                                 "resampling_strategy":'cv',
                                                 "resampling_strategy_arguments":{'folds': 3}, "n_jobs":4,
                                                 "ensemble_size":5, "ensemble_nbest":10, "ml_memory_limit":30000})
    #configs["gb"] = get_base_config()
    #configs["knn-25"] = get_base_config(model_fn=KNeighborsClassifier,
    #                                                  model_params={"n_neighbors": 25})
    #configs["knn-50"] = get_base_config(model_fn=KNeighborsClassifier,model_params={"n_neighbors": 50, "n_jobs":-1 })
    #configs["knn-100"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 100})
    #configs["knn-200"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 200})
    #configs["knn-300"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 300})
    #configs["knn-500"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 500})
    #configs["knn-1000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 1000})
    #configs["knn-2000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 2000})
    #configs["nb"] = get_base_config(model_fn=ComplementNB)
    #configs["nb-Compliment"] = get_base_config(model_fn=ComplementNB)
    #configs["nb-Multi"] = get_base_config(model_fn=MultinomialNB)
    #configs["random stratified"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "stratified"})
    #configs["random uniform"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "uniform"})
    #configs["rf"] = get_rf_baseline_config()
    #configs["xgboost"] = get_xgboost_baseline_config()
    return configs


def str_to_year(x):
    return int(x[0:4]) if isinstance(x,str) else np.nan


def get_default_index(key="id"):
    if key=="id":
        return {"person": ["person_id","year_of_birth", "gender_concept_id", "race_concept_id"],
                "condition_occurrence": ["condition_concept_id"],
                "procedure_occurrence": ["procedure_concept_id"],
                "measurement": ["measurement_concept_id"],
                "drug_exposure": ["drug_concept_id"]}
    elif key=="dates":
        return {"condition_occurrence": [("condition_concept_id", ["condition_start_date","person_id",str_to_year ])],
                "procedure_occurrence": [("procedure_concept_id", ["procedure_date", "person_id", str_to_year])],
                "measurement": [("measurement_concept_id", ["measurement_date", "person_id", str_to_year])],
                "drug_exposure": [("drug_concept_id", ["drug_exposure_start_date", "person_id", str_to_year])],
                "person": ["person_id","year_of_birth", "gender_concept_id", "race_concept_id"]}
def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config
