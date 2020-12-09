import pandas as pd
import numpy as np
import os
import sys
sys.path.extend(['/'])
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
import time
import app.models as model_includes
import app.model_configs as model_configs
from sklearn.ensemble import AdaBoostClassifier
import warnings
import os
import catboost as ct
import sklearn as sk


#RUN different models
model_names = ["catboost", "ada"]

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
if config["feature importance"] and isinstance(config_select["model"], AdaBoostClassifier):
    importances = config_select["model"].feature_importances_
elif config["feature importance"] and isinstance(config_select["model"], ct.CatBoostClassifier):

    config_base = list(configs.values())[0]
    date_lags = config_base["date lags"]

    data_sp, labels_iter = model_includes.get_sparse_person_features_mat(data)
    x_train, x_test, y_train, y_test, keys_train, keys_test = sk.model_selection.train_test_split(data_sp, list(labels_iter.values()), list(labels_iter.keys()), train_size=config_base["train size"])

    importances = config_select["model"].get_feature_importance(type=config["feature importance method"],
                                                                data=model_configs.ct.Pool(x_train, y_train))
    if config["feature importance method"] == "ShapValues":
        importances = np.mean(importances, axis=0)

aa = np.transpose(np.vstack(
    ([int(v) for k, v in list(uids.keys())], [int(v) for k, v in list(uids.keys())])))
pd.DataFrame(aa).to_csv(config["output path"] + "features.csv", header=None, index=None)
print(config["output path"] + "features.csv")
if importances is not None:
    pd.DataFrame(importances).to_csv(config["output path"] + "feature_weights.csv", header=None, index=None)
    print(config["output path"] + "feature_weights.csv")
model_configs.pickle_nms(config_select, config["model path"] + "config.joblib")

print("Total time:" + str(time.time()-t), flush=True)

