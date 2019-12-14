

import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import os
import sys
sys.path.extend(['/'])
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
import joblib as jl
from app import models as model_includes
from app import configs as model_configs
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

t = time.time()
config_paths = model_configs.get_base_config()
config = jl.load(config_paths["model path"] + "config.joblib")
tt = time.time()
data = model_includes.read_ehrdc_data(config["test path"])
print("Data load time:" + str(time.time() - tt))
if config["model name"] == "static demographic baseline":
    p = model_includes.model_static_patient_predict(data, config["model"])
    p.to_csv(config_paths["output path"]+ "predictions.csv")
elif config["model name"] == "static uid model selection":
    uids = jl.load(config_paths["scratch path"] + "uids.joblib")
    if "cv split key" in config and config["cv split key"] == "dates":
        p = model_includes.model_sparse_feature_test(data, config, uids=uids, split_key=config["cv split key"], date_lag=config["date lag"])
    else:
        p = model_includes.model_sparse_feature_test(data, config, uids=uids)
    p.to_csv(config_paths["output path"] + "predictions.csv")
print("total time:" + str(time.time()-t))