import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c

import sklearn as sk
import time
import joblib as jl
import includes.models as model_includes
import includes.configs as model_configs

t = time.time()
config = model_configs.get_rf_baseline_config()
config["model name"] = "static model selection"

data = model_includes.read_ehrdc_data(config["train path"])
if "train" in config and config["train"]:
    print("Running: " + config["model name"])
    if config["model name"] == "static baseline":
        m = model_includes.model_static_patient_train(data, data["death"]["label"], config)
        jl.dump(m, config["model path"] + "gb_static_patient_gb.joblib")
    elif config["model name"] == "static model selection":
        configs = model_configs.get_baseline_cv_configs()
        model, selected, perf, metrics_out, configs = model_includes.model_sparse_feature_cv(data, configs, iters=10)
        print("Selected: " + selected)
        print(perf)
        jl.dump(model, config["model path"] + "static_modelselection.joblib")
print(time.time() - t)
