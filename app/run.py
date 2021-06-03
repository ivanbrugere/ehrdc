import datetime
import os
import sys
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
sys.path.append(os.getcwd())
import time
import models as model_includes
import model_configs as model_configs
import warnings
import os
import joblib as jl
from pathlib import Path

model_names = ["logistic", "adaboost", "catboost"]
config_params = {"load_model": True,
                 "output_path": os.path.join(Path(os.getcwd()).parent, "output"),
                 "k_folds": 10}

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
t = time.time()
configs = model_configs.get_baseline_cv_configs(model_names=model_names, config_params=config_params)
config = list(configs.values())[0]
tt = time.time()

if not os.path.exists(config["output path"]):
    os.makedirs(config["output path"])

data = model_includes.read_ehrdc_data(config["train npy"])
data_eval = model_includes.read_ehrdc_data(config["test npy"])
configs = model_configs.get_baseline_cv_configs(model_names=model_names)

selected_models, importances, importances_mat, preds, performance  = model_includes.model_selection_and_evaluation(data, data_eval, configs)

t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if importances_mat is not None:
    jl.dump(importances_mat, os.path.join(config["output path"], "shap_matrix_{}.gz".format(t)))
importances.to_csv(os.path.join(config["output path"], "feature_weights_{}.csv".format(t)))
preds.to_csv(os.path.join(config["output path"], "predictions_{}.csv".format(t)))
jl.dump(performance, os.path.join(config["output path"], "performance_{}.gz".format(t)))