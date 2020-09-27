

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
from app import model_configs as model_configs
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


covid = True

if covid:
    pipeline_vars = {"pipeline": "covid"}
else:
    pipeline_vars = {"pipeline": ""}


t = time.time()
config_paths = model_configs.get_base_config(**pipeline_vars)
config = model_configs.unpickle_nms(config_paths["model path"] + "config.joblib")
#config = jl.load(config_paths["model path"] + "config.joblib")
tt = time.time()
if "train npy" in config and os.path.isdir(config["test npy"]["path"]):
    data = model_includes.read_ehrdc_data(config["test npy"], **pipeline_vars)
else:
    data = model_includes.read_ehrdc_data(config["test path"], **pipeline_vars)
print("Data load time:" + str(time.time() - tt), flush=True)
if config["model name"] == "static demographic baseline":
    p = model_includes.model_static_patient_predict(data, config["model"])
    p.to_csv(config_paths["output path"]+ "predictions.csv")
elif config["model name"] == "static uid model selection":
    uids = config["uids"]
    if "cv split key" in config and config["cv split key"] == "dates":
        p = model_includes.model_sparse_feature_test(data, config, uids=uids, split_key=config["cv split key"], date_lag=config["date lag"])
    else:
        p = model_includes.model_sparse_feature_test(data, config, uids=uids)
    p.to_csv(config_paths["output path"] + "predictions.csv")
print("total time:" + str(time.time()-t), flush=True)

