import pandas as pd
import numpy as np
import os
import sys
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
sys.path.append(os.getcwd())
import time
import models as model_includes
import model_configs as model_configs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
import os
import catboost as ct
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap
#RUN different models
model_names = ["logistic", "ada", "catboost"]
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
t = time.time()
configs = model_configs.get_baseline_cv_configs(model_names=model_names)
config = list(configs.values())[0]
tt = time.time()

if not os.path.exists(config["output path"]):
    os.makedirs(config["output path"])
if not os.path.exists(config["scratch path"]):
    os.makedirs(config["scratch path"])
if not os.path.exists(config["model path"]):
    os.makedirs(config["model path"])

data = model_includes.read_ehrdc_data(config["train npy"])
configs = model_configs.get_baseline_cv_configs(model_names=model_names)

config_select, selected, perf, metrics_out, configs, uids = model_includes.model_sparse_feature_cv_train(data, configs)
config_select["uids"] = uids
importances = None
config_base = list(configs.values())[0]
date_lags = config_base["date lags"]

data_sp, labels_iter = model_includes.get_sparse_person_features_mat(data)
x_train, x_test, y_train, y_test, keys_train, keys_test = sk.model_selection.train_test_split(data_sp, list(
    labels_iter.values()), list(labels_iter.keys()), train_size=config_base["train size"])

if config["feature importance"] and isinstance(config_select["model"], AdaBoostClassifier):
    importances = config_select["model"].feature_importances_
elif config["feature importance"] and isinstance(config_select["model"], (ct.CatBoostClassifier)): #model_configs.PairedKnn, model_configs.NNC, KNeighborsClassifier)):
    shap_values = shap.Explainer(config_select["model"])(x_train)
    importances = np.mean(shap_values.values, axis=0)
# elif config["feature importance"] and isinstance(config_select["model"], (KNeighborsClassifier)):
#     kn = shap.KernelExplainer(config_select["model"].predict, shap.kmeans(x_train, 100))
#     importances = np.mean(kn.shap_values(x_test).values, axis=0)
elif config["feature importance"] and isinstance(config_select["model"], (LogisticRegression)):
    importances = config_select["model"].coef_.flatten()
aa = np.transpose(np.vstack(
    ([int(v) for k, v in list(uids.keys())], [int(v) for k, v in list(uids.keys())])))
pd.DataFrame(aa).to_csv(config["output path"] + "features.csv", header=None, index=None)
print(config["output path"] + "features.csv")
if importances is not None:
    pd.DataFrame(importances).to_csv(config["output path"] + "feature_weights.csv", header=None, index=None)
    print(config["output path"] + "feature_weights.csv")
model_configs.pickle_nms(config_select, config["model path"] + "config.joblib")

print("Total time:" + str(time.time()-t), flush=True)

