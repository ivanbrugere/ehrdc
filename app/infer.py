import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib as jl
import includes.models as model_includes
import includes.configs as model_configs

config = model_configs.get_gb_baseline_config()
m = jl.load(config["model path"] + "gb_static_patient_gb.joblib")
data_test, _, _ = model_includes.read_ehrdc_data(config["test path"])
p = model_includes.model_static_patient_predict(data_test, m)
p.to_csv(config["output path"]+ "predictions.csv")