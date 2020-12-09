import pandas as pd
import numpy as np
import collections as c
import os
import operator
import sklearn as sk
import scipy.sparse as sp
import time
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')

from app import model_configs as model_configs

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
    if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
        p, _ = evaluate_paired_model(config, data_sp)
    else:
        p = config["model"].predict_proba(data_sp)
    print("Finished inference", flush=True)

    keys_iter = pd.Series(list(p_ids.keys()), name="person_id")

    print("Inference time:" + str(time.time() - t), flush=True)
    p[p < 0] = 0
    p[p>1] = 1
    p[np.isnan(p)] = 0
    return pd.DataFrame(p[:, 1], index=keys_iter, columns=["score"])

def model_sparse_feature_cv_train(data, configs):
    t = time.time()
    config_base = list(configs.values())[0]
    date_lag = [0]

    iters = config_base["cv iters"]
    uids_feats = dict(zip(zip(range(data["x"].shape[1]), range(data["x"].shape[1])), range(data["x"].shape[1])))



    metrics_out = c.defaultdict(list)
    print("Data build time: " + str(time.time() - t), flush=True)
    tt = time.time()

    labels_store = {}

    data_sp, labels_iter = get_sparse_person_features_mat(data)
    labels_store[tuple(date_lag)] = labels_iter
    for ii in range(iters):
        config_base = model_configs.get_default_train_test(config_base)
        x_train, x_test, y_train, y_test, keys_train, keys_test = sk.model_selection.train_test_split(data_sp, list(labels_iter.values()), list(labels_iter.keys()), train_size=config_base["train size"])
        print("CV Train data: " + str((x_train.shape, x_train.nnz, x_train.dtype)), flush=True)
        print("CV Test data: " + str((x_test.shape, x_test.nnz, x_test.dtype)), flush=True)

        for key_c, config in configs.items():
            model_configs.reset_model(config["model"])
        x_apps_train = []
        x_apps_val = []
        for jj, (key_c, config) in enumerate(configs.items()):
            key_c = (key_c, tuple(date_lag))
            print("(Model,iteration, %): " + str((key_c, ii, jj/len(configs))), flush=True)

            #train
            ttt = time.time()
            if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
                x_apps_train = train_paired_model(config, x_train, np.array(y_train), x_apps_train)
            elif isinstance(config["model"], model_configs.NNC):
                config["model"].train_nn(x_train, np.array(y_train))
            else:
                config["model"].fit(x_train, np.array(y_train))
            print("CV Train: " + str(time.time() - ttt), flush=True)

            #eval
            ttt = time.time()
            if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
                y_pred, x_apps_val = evaluate_paired_model(config, x_test, x_apps_val)
            elif (isinstance(config["model"], model_configs.PairedKnn) or isinstance(config["model"], model_configs.PairedPipeline)) and isinstance(config["model"].f_rep, model_configs.NNC) and hasattr(config["model"].f_rep, "module_"):
                if config["model"].f_rep.module_.cache is not None:
                    inds_iter = config["model"].f_rep.module_.cache
                else:
                    inds_iter = None
                y_pred = config["model"].predict_proba(x_test, inds=inds_iter)
            else:
                y_pred = config["model"].predict_proba(x_test)
            print("CV Predict: " + str(time.time() - ttt), flush=True)
            metrics_out[key_c].append(sk.metrics.roc_auc_score(y_test, y_pred[:, 1]))
    perf = {k: {"mean": np.mean(v), "std": np.std(v)} for k,v in metrics_out.items()}
    selected, selected_mean = sorted({k:v["mean"] for k,v in perf.items()}.items(), key=operator.itemgetter(1))[::-1][0]
    config_select = configs[selected[0]]
    config_select["date lag"] = selected[1]
    print("Selected: " + str(selected),flush=True)
    print(perf,flush=True)
    print("Training full selected model",flush=True)
    model_configs.reset_model(config_select["model"])
    if isinstance(config_select["model"], dict) and "nets" in config_select["model"] and "pred" in config_select["model"]:
        train_paired_model(config_select, data_sp, np.array(list(labels_store[selected[1]].values())))
    elif isinstance(config_select["model"], model_configs.NNC):
        config_select["model"].train_nn(data_sp, np.array(list(labels_store[selected[1]].values())))
    else:
        config_select["model"].fit(data_sp, np.array(list(labels_store[selected[1]].values())))

    config_select["train shape"] = data_sp.shape
    print("Model cv time: " + str(time.time() - tt), flush=True)
    return config_select, selected, perf, metrics_out, configs, uids_feats


def evaluate_paired_model(config_select, data_sp, y_label=None, train=False, x_apps=[]):
    for k,vm in config_select["model"]["nets"].items():
        vm.train_nn(data_sp, y_label)
        if not x_apps:
            y_net_pred = vm.predict_proba(data_sp)[:, 1]
            x_apps.append(sp.csr_matrix((y_net_pred, (range(len(y_net_pred)), np.zeros(len(y_net_pred)))),
                                  shape=(len(y_net_pred), 1)))
    data_sp_iter = sp.hstack([data_sp] + x_apps).tocsr()
    if train:
        config_select["model"]["pred"].fit(data_sp_iter, y_label)
        return x_apps
    else:
        return config_select["model"]["pred"].predict_proba(data_sp_iter), x_apps

def train_paired_model(config_select, data_sp, y_label, x_apps=[]):
    return evaluate_paired_model(config_select, data_sp, y_label=y_label, train=True, x_apps=x_apps)

def get_sparse_person_features_mat(data_np):
    return sp.csr_matrix(data_np["x"]), dict(zip(range(len(data_np["y"])), data_np["y"]))
