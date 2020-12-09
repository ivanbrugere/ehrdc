import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.exceptions import NotFittedError
import joblib as jl
from pathlib import Path
import catboost as ct
os.environ["OMP_NUM_THREADS"] = "8"

def dummy_ret(x):
    return x

def reset_model(m):
    if isinstance(m, dict) and "nets" in m:
        for k, vm in m["nets"].items():
            reset_model(vm)
    if isinstance(m, dict) and "preds" in m:
        reset_model(m["preds"])
    # elif isinstance(m, (NNC, PairedKnn, PairedPipeline, LargeEnsemble)):
    #     m.reset()
    elif isinstance(m, AdaBoostClassifier):
        m = sk.base.clone(m)
    else:
        m.__init__(**m.get_params())

def is_fitted(m, data):
    try:
        m.predict_proba(data[0:1, :])
        return 1
    except (NotFittedError):#, xgb.core.XGBoostError):
        return 0

def pickle_nms(config, file):
    jl.dump(config, file)

def unpickle_nms(file):
    config = jl.load(file)
    return config

def get_base_config(model_fn=None, model_params={}, name=None):
    config = {}
    if name is None:
        config["model name"] = "static uid model selection"
    else:
        config["model name"] = name

    if model_fn is None:
        config["model_fn"] = GradientBoostingClassifier
    else:
        config["model_fn"] = model_fn
    config["model_params"] = model_params
    config["model"] = config["model_fn"](**model_params)
    prefix = Path(os.getcwd()).parent
    config["model path"] = os.path.join(prefix, "model", "")
    config["output path"] = os.path.join(prefix, "output", "")
    config["scratch path"] = os.path.join(prefix, "scratch", "")
    config["train npy"] = {"path": os.path.join("..", "train", ""), "map": {"negative.npy": 0, "positive.npy": 1}, "fields": {"data": "x", "labels":"y"}}
    config["test npy"] = {"path": os.path.join("..", "test", ""), "map": {"negative.npy": 0, "positive.npy": 1}, "fields": {"data": "x", "labels": "y"}}
    config["train"] = True
    config["do cv"] = True
    config["date lags"] = [[0]]
    config["join field"] = "person_id"
    config["cv iters"] = 5
    config["cv split key"] = "id"
    config["feature importance"] = True
    config["feature importance method"] = "FeatureImportance"
    #config["feature importance method"] = "ShapValues"
    return config

def get_baseline_cv_configs(model_names=["catboost"]):
    configs = dict()
    if "ada" in model_names:
        depths = [1, 2, 3, 4]
        ns = [250, 500]
        lrs = [0.1, 0.25, 0.5, 1]
        for d in depths:
            for lr in lrs:
                for n in ns:
                    par = {"n_estimators": n, "learning_rate": lr, "base_estimator": DecisionTreeClassifier(max_depth=d)}
                    configs[("adaboost", d, lr, n)] = get_base_config(model_fn=AdaBoostClassifier,model_params=par)
    if "catboost" in model_names:

        depths = [7]
        objectives = ["Logloss", "CrossEntropy"]
        ns = [250, 500]
        lrs = [0.01, 0.05]
        l2_leaf_regs = [1, 3, 6]
        rsms = [.5, 1]
        class_weights = [None, "Balanced"]

        for d in depths:
            for o in objectives:
                for lr in lrs:
                    for l2 in l2_leaf_regs:
                        for rsm in rsms:
                            for n in ns:
                                for w in class_weights:
                                    par = {"logging_level": "Silent", "n_estimators": n, "depth": d, "loss_function": o,
                                     "learning_rate": lr, "l2_leaf_reg": l2, "rsm": rsm, "auto_class_weights": w}
                                    if o == "CrossEntropy":
                                        w = ()
                                        del par["auto_class_weights"]
                                    configs[("catboost", d, o, lr, l2, rsm, w, n)] = get_base_config(model_fn=ct.CatBoostClassifier, model_params=par)
                                    if o == "CrossEntropy":
                                        break

    print("Models #: " + str(len(configs)))
    return configs

def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config