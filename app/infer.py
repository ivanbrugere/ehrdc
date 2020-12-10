import os
import sys
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')
sys.path.append(os.getcwd())
import models as model_includes
import model_configs as model_configs
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

t = time.time()
config_paths = model_configs.get_base_config()
config = model_configs.unpickle_nms(config_paths["model path"] + "config.joblib")

data = model_includes.read_ehrdc_data(config["test npy"])

p = model_includes.model_sparse_feature_test(data, config)
p.to_csv(config_paths["output path"] + "predictions.csv")
print("total time:" + str(time.time()-t), flush=True)