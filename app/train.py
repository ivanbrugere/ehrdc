import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c

import sklearn as sk
import os
import sys
sys.path.extend(['/'])
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
import time
import joblib as jl
import app.models as model_includes
import app.model_configs as model_configs

import warnings
import os
import glob
import catboost as ct
covid = True

model_names = ["embed", "embed-knn", "ada", "catboost"]

if covid:
    pipeline_vars = {"pipeline": "covid"}
else:
    pipeline_vars = {"pipeline": ""}

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
t = time.time()
split_key = "id"
configs = model_configs.get_baseline_cv_configs(**pipeline_vars, model_names=model_names)
config = list(configs.values())[0]
load_only=False
tt = time.time()

if not os.path.exists(config["output path"]):
    os.makedirs(config["output path"])
if not os.path.exists(config["scratch path"]):
    os.makedirs(config["scratch path"])
if not os.path.exists(config["model path"]):
    os.makedirs(config["model path"])


if "train npy" in config and os.path.isdir(config["train npy"]["path"]):
    data = model_includes.read_ehrdc_data(config["train npy"], **pipeline_vars)
else:
    data = model_includes.read_ehrdc_data(config["train path"], **pipeline_vars)
print("Data load time:" + str(time.time()-tt))
if "train" in config and config["train"]:
    print("Running: " + config["model name"] + "," + config["cv split key"])
    if config["model name"] == "static demographic baseline":
        config_trained = model_includes.model_static_patient_train(data, data["death"]["label"], config)
        jl.dump(config_trained, config["model path"] + "config.joblib")
    elif config["model name"] == "static uid model selection":
        configs = model_configs.get_baseline_cv_configs(**pipeline_vars, model_names=model_names)
        if load_only:

            x_train, x_test, y_train, y_test, keys_train, keys_test = model_includes.preprocess_data(data, configs, split_key="id")
        else:
            config_select, selected, perf, metrics_out, configs, uids = model_includes.model_sparse_feature_cv_train(data, configs, split_key=split_key)
            config_select["uids"] = uids
            model_configs.pickle_nms(config_select, config["model path"] + "config.joblib")

            if(config["feature importance"] and isinstance(config_select["model"], ct.CatBoostClassifier) ):
                x_train, x_test, y_train, y_test, keys_train, keys_test = model_includes.preprocess_data(data, configs,split_key="id")

                importances = config_select["model"].get_feature_importance(type=config["feature importance method"],
                                                                 data=model_configs.ct.Pool(x_train, y_train))
                if config["feature importance method"] == "ShapValues":
                    importances = np.mean(importances, axis=0)

                aa = np.transpose(np.vstack(
                    ([int(v) for k, v in list(uids.keys())], [int(v) for k, v in list(uids.keys())])))
                pd.DataFrame(aa).to_csv(config["output path"]+ "features.csv", header=None, index=None)
                print(config["output path"]+ "features.csv")
                pd.DataFrame(importances).to_csv(config["output path"]+ "feature_weights.csv", header=None, index=None)
                print(config["output path"]+ "feature_weights.csv")
            #jl.dump(config_select, config["model path"] + "config.joblib")
            # jl.dump(uids, config["model path"] + "uids.joblib")
            # print(config["model path"] + "uids.joblib")
            # print(os.path.getsize(config["model path"] + "uids.joblib"))
            #del config, configs, data, perf, selected, uids, config_select, metrics_out
print("Total time:" + str(time.time()-t), flush=True)

