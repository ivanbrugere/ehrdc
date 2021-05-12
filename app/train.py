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
model_names = ["logistic", "adaboost", "catboost"]
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
t = time.time()
configs = model_configs.get_baseline_cv_configs(model_names=model_names)
config = list(configs.values())[0]
tt = time.time()

if not os.path.exists(config["output path"]):
    os.makedirs(config["output path"])

data = model_includes.read_ehrdc_data(config["train npy"])
data_eval = model_includes.read_ehrdc_data(config["test npy"])
configs = model_configs.get_baseline_cv_configs(model_names=model_names)

selected_models, importances, preds  = model_includes.model_selection_and_evaluation(data, data_eval, configs)
importances.to_pickle(config["output path"] + "feature_weights.gz")
preds.to_pickle(config["output path"] + "predictions.gz")
jl.dump(selected_models, config["output path"] + "models.gz")