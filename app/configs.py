import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB #ComplementNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import os
import numpy as np
from autosklearn import classification as ask
import shutil

os.environ["OMP_NUM_THREADS"] = "8"


class LDA_classifier:
    def __init__(self, **kwargs):
        self.model = sk.decomposition.LatentDirichletAllocation(**kwargs)
        self.weights = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.weights = self.get_factor_risk_vector(X, y)

    def predict_proba(self, X):
        return self.evaluate_factor_risk(X)

    def get_factor_risk_vector(self, x, y):
        y_int = [-1 if not x else 1 for x in y]
        x_tran = self.model.transform(x)
        aa = np.sum(x_tran * np.repeat(np.expand_dims(np.array(y_int), 1), x_tran.shape[1], axis=1), axis=0)
        return 1 - (aa - max(aa)) / (abs(max(aa)) - abs(min(aa)))

    def evaluate_factor_risk(self, x):
        x_tran = self.model.transform(x)
        risks = np.sum(np.repeat(np.expand_dims(self.weights, 0), x_tran.shape[0], axis=0) * x_tran, axis=1)
        return np.vstack((np.zeros(len(risks)), risks)).transpose()


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
    config["train path"] = "../train/"
    config["test path"] = "../infer/"
    config["model path"] = "../model/"
    config["output path"] = "../output/"
    config["scratch path"] = "../scratch/"
    config["filter path"] = "./filter/"
    config["train"] = True
    config["do cv"] = True
    config["date lags"] = [[0]]
    config["join field"] = "person_id"
    config["cv iters"] = 1
    config["cv split key"] = "id"
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
    # configs["auto"] = get_base_config(model_fn=ask.AutoSklearnClassifier, model_params={"time_left_for_this_task":1500, "per_run_time_limit":300,
    #                                              "n_jobs":8,
    #                                              "ensemble_size":3, "ensemble_nbest":20, "ml_memory_limit":30000})
    #
    # tmp_path = configs["auto"]["model path"] + "tmp"
    # if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
    #     shutil.rmtree(tmp_path)
    # out_path = configs["auto"]["model path"] + "out"
    # if os.path.exists(out_path) and os.path.isdir(out_path):
    #     shutil.rmtree(out_path)
    # configs["auto"]["model"].tmp_folder = configs["auto"]["model path"] + "tmp"
    # configs["auto"]["model"].output_folder = configs["auto"]["model path"] + "out"
    # #


    # configs["LDA-10"] = get_base_config(model_fn=LDA_classifier, model_params={"learning_method":"online", "batch_size":1000, "n_jobs":-1,
    #                                                                            "n_components":10})
    # configs["LDA-20"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 20})
    # configs["LDA-30"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 30})
    # configs["LDA-40"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 40})
    # configs["LDA-50"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 50})



    configs["gb"] = get_base_config()
    #configs["knn-25"] = get_base_config(model_fn=KNeighborsClassifier,
    #                                                  model_params={"n_neighbors": 25})
    #configs["knn-50"] = get_base_config(model_fn=KNeighborsClassifier,model_params={"n_neighbors": 50, "n_jobs":-1 })
    #configs["knn-100"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 100})
    #configs["knn-200"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 200})
    #configs["knn-300"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 300})
    #configs["knn-500"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 500})
    #configs["knn-1000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 1000})
    #configs["knn-2000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 2000})
    configs["nb"] = get_base_config(model_fn=BernoulliNB)
    #configs["nb-Compliment"] = get_base_config(model_fn=ComplementNB)
    #configs["nb-Multi"] = get_base_config(model_fn=MultinomialNB)
    #configs["random stratified"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "stratified"})
    #configs["random uniform"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "uniform"})
    configs["rf"] = get_rf_baseline_config()

    p = {"max_depth": 12, "n_jobs:": 8, "n_estimators": 150}
    objectives = ["binary:logistic", "reg:squarederror"]
    alphas = [0, 0.5, 1]
    lambdas = [0, 0.5, 1]
    boosters = ["gbtree", "dart", "gblinear"]
    trees = ["auto", "hist"]

    for o in objectives:
        for a in alphas:
            for l in lambdas:
                if a != l:
                    for b in boosters:
                        for t1 in trees:
                            p2 = p.copy()
                            p2["booster"] = b
                            p2["reg_lambda"] = l
                            p2["reg_alpha"] = a
                            p2["objective"] = o
                            p2["tree_method"] = t1
                            configs[("xgboost", o, a, l, b, t1)] = get_xgboost_baseline_config(model_params=p2)
    return configs


def str_to_year(x):
    return int(x[0:4]) if isinstance(x,str) else np.nan


def get_default_index(key="id"):
    if key=="id":
        return {"condition_occurrence": ["condition_concept_id"],
                "procedure_occurrence": ["procedure_concept_id"],
                "observation":["observation_concept_id"],
                "measurement": ["measurement_concept_id"],
                "drug_exposure": ["drug_concept_id"],
                "person": ["year_of_birth", "gender_concept_id", "race_concept_id", "person_id"]}
    elif key=="dates":
        return {"condition_occurrence": [("condition_concept_id", ["condition_start_date","person_id",str_to_year ])],
                "procedure_occurrence": [("procedure_concept_id", ["procedure_date", "person_id", str_to_year])],
                "measurement": [("measurement_concept_id", ["measurement_date", "person_id", str_to_year])],
                "drug_exposure": [("drug_concept_id", ["drug_exposure_start_date", "person_id", str_to_year])],
                "person": ["year_of_birth", "gender_concept_id", "race_concept_id", "person_id"]}
def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config
