import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib as jl
import includes.models as model_includes
import includes.configs as model_configs

config_paths = model_configs.get_base_config()
config = jl.load(config_paths["model path"] + "config.joblib")
if config["model name"] == "static demographic baseline":
    data_test, _, _ = model_includes.read_ehrdc_data(config["test path"])
    p = model_includes.model_static_patient_predict(data_test, config["model"])
    p.to_csv(config_paths["output path"]+ "predictions.csv")
elif config["model name"] == "static uid model selection":
    jl.load(config_paths["scratch path"] + "uids.joblib")
    data_test, _, _ = model_includes.read_ehrdc_data(config["test path"])
    p = model_includes.model_static_patient_predict(data_test, config["model"])
    p.to_csv(config_paths["output path"] + "predictions.csv")