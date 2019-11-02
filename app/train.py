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
import app.configs as model_configs

t = time.time()
config = model_configs.get_rf_baseline_config()
config["model name"] = "static uid model selection"
config["cv iters"] = 2
config["cv split key"] = "dates"
config["cv date lags"] = [[0], [0, 1], [0, 1, 2]]
tt = time.time()
data = model_includes.read_ehrdc_data(config["train path"])
print("Data load time:" + str(time.time()-tt))
if "train" in config and config["train"]:
    print("Running: " + config["model name"] + "," + config["cv split key"])
    if config["model name"] == "static demographic baseline":
        config_trained = model_includes.model_static_patient_train(data, data["death"]["label"], config)
        jl.dump(config_trained, config["model path"] + "config.joblib")
    elif config["model name"] == "static uid model selection":
        configs = model_configs.get_baseline_cv_configs()
        config_select, selected, perf, metrics_out, configs, uids = model_includes.model_sparse_feature_cv_train(data, configs, iters=config["cv iters"], split_key=config["cv split key"], date_lags=config["cv date lags"])
        print("Selected: " + str(selected))
        print(perf)
        jl.dump(config_select, config["model path"] + "config.joblib")
        jl.dump(uids, config["scratch path"] + "uids.joblib")
print("Total time:" + str(time.time()-t))