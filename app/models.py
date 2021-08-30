import pandas as pd
import numpy as np
import collections as c
import operator
import scipy.sparse as sp
import time
from sklearn.ensemble import AdaBoostClassifier
import os
import catboost as ct
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import shap
import joblib as jl

if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')

from sklearn.model_selection import KFold


import model_configs as model_configs

def read_ehrdc_data(path):
    print("Data path: " + str(path), flush=True)
    if isinstance(path, dict):
        xs = []
        ys = []
        for k,v in path["map"].items():
            x = np.load(os.path.join(path["path"], k))
            y = np.ones((x.shape[0], 1))*v
            xs.append(x)
            ys.append(y)

        return {path["fields"]["data"]: np.vstack(xs), path["fields"]["labels"]:np.transpose(np.vstack(ys))[0]}

def model_sparse_feature_test(data, config):
    t = time.time()

    p_ids = dict(zip(range(data["x"].shape[0]), range(data["x"].shape[0])))

    data_sp, labels_iter = get_sparse_person_features_mat(data)
    p = config["model"].predict_proba(data_sp)
    print("Finished inference", flush=True)

    keys_iter = pd.Series(list(p_ids.keys()), name="person_id")

    p[p < 0] = 0
    p[p>1] = 1
    p[np.isnan(p)] = 0
    print("Inference time: " + str(time.time() - t), flush=True)
    return pd.DataFrame(p[:, 1], index=keys_iter, columns=["score"])
def get_importances(data, config_select):
    importances_full = None
    if isinstance(config_select["model"], AdaBoostClassifier):
        importances = config_select["model"].feature_importances_
    elif isinstance(config_select["model"],ct.CatBoostClassifier):
        shap_values = shap.Explainer(config_select["model"])(data)
        importances = np.mean(shap_values.values, axis=0)
        importances_full = pd.DataFrame(shap_values.values)
    elif isinstance(config_select["model"], LogisticRegression):
        importances = config_select["model"].coef_.flatten()
    return pd.DataFrame(importances, index=pd.Series(range(data.shape[1]), name="feature_id"), columns=["importance"]), importances_full

def model_selection_and_evaluation(data, data_eval, configs):
    config_base = list(configs.values())[0]
    model_path = os.path.join(config_base["output path"], "model.gz")
    metrics_out = c.defaultdict(list)

    if config_base["load model"] and os.path.exists(model_path):
        config_select = jl.load(model_path)
        print("LOADED saved model: {}, {}".format(model_path, config_select["model"]))
    else:
        data_sp, labels_iter = get_sparse_person_features_mat(data)
        kf = KFold(n_splits=config_base["k folds"], shuffle=True)
        X = data_sp
        y = np.array(list(labels_iter.values()))

        for ii, (idx_train, idx_test) in enumerate(kf.split(data_sp, list(labels_iter.values()), list(labels_iter.keys()))):
            print("Split:{}/{}".format(ii, config_base["k folds"]))
            x_train, x_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
            tt = time.time()

            for key_c, config in configs.items():
                model_configs.reset_model(config["model"])
            for jj, (key_c, config) in enumerate(configs.items()):
                #train
                config["model"].fit(x_train, np.array(y_train))
                #eval
                y_pred = config["model"].predict_proba(x_test)
                metrics_out[key_c].append(sk.metrics.roc_auc_score(y_test, y_pred[:, 1]))
            print("Fold time: {}".format(time.time() - tt))
        perf = {k: {"mean": np.mean(v), "std": np.std(v)} for k,v in metrics_out.items()}
        selected, selected_mean = sorted({k:v["mean"] for k,v in perf.items()}.items(), key=operator.itemgetter(1))[::-1][0]
        config_select = configs[selected]

    p = model_sparse_feature_test(data_eval, config_select)
    importances, importances_mat = get_importances(data_eval["x"], config_select)

    return config_select, importances, importances_mat, p, metrics_out

def get_sparse_person_features_mat(data_np):
    return sp.csr_matrix(data_np["x"]), dict(zip(range(len(data_np["y"])), data_np["y"]))
