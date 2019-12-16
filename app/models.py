import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c
import os
import sys
import operator
import sklearn as sk
import joblib as jl
import scipy.sparse as sp
import itertools as it
import datetime
import time
import torch
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
import skorch
#from autosklearn import classification as ask
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'/app')

from app import configs as model_configs


def read_ehrdc_data(path, data_keys=None, label_keys=None, useful_keys=None, expand_labels=("person", "person_id", "death_date"),
                    filter_useful=True, apply_label_fn=model_configs.str_to_year):
    print("Data path: " + str(path))
    if data_keys is None:
        data_keys = ["condition_occurrence", "drug_exposure", "measurement", "observation_period", "observation",
                     "person", "procedure_occurrence", "visit_occurrence"]
    if label_keys is None:
        label_keys = ["death"]

    print(os.getcwd())
    with open("OMOP_useful_columns.csv") as f:
        useful_keys = c.defaultdict(list)
        dd = csv.DictReader(f)
        for r in dd:
            if filter_useful and r["Useful"] == "TRUE":
                useful_keys[r["TabNam"]].append(r["ColNam"])
            elif not filter_useful:
                useful_keys[r["TabNam"]].append(r["ColNam"])
    data = {}
    labels = {}
    for k in data_keys:
        f = path + k + ".csv"
        if pathlib.Path(f).exists():
            useful_keys[k] = list(set(useful_keys[k]).intersection(set(pd.read_csv(f, nrows=0).columns)))
            data[k] = pd.read_csv(f, usecols=useful_keys[k])
            data[k] = data[k].dropna(axis=1, how="all")
    for k in label_keys:
        f = path + k + ".csv"
        if pathlib.Path(f).exists():
            useful_keys[k] = list(set(useful_keys[k]).intersection(set(pd.read_csv(f, nrows=0).columns)))
            labels[k] = pd.read_csv(path + k + ".csv", usecols=useful_keys[k])
            if expand_labels:
                positives = set(labels[k][expand_labels[1]])
                expand_values = {k:v for k,v in zip(labels[k][expand_labels[1]], labels[k][expand_labels[2]])}
                inds = data[expand_labels[0]][expand_labels[1]]
                a = pd.Series([apply_label_fn(expand_values[i]) if i in positives else np.nan for i in inds])
                b = pd.Series([int(i in positives) for i in inds])
                data[k] = pd.concat([inds, b, a], axis=1, keys=[inds.name, "label", expand_labels[2]])

    return data


def model_static_patient_preprocess(data_train):
    pop_fields = ['day_of_birth', 'time_of_birth']
    one_hot_fields = ['gender_concept_id',
                      'race_concept_id',
                      'ethnicity_concept_id', 'location_id', 'provider_id', 'care_site_id',
                      'person_source_value', 'gender_source_value',
                      'gender_source_concept_id', 'race_source_value',
                      'race_source_concept_id', 'ethnicity_source_value',
                      'ethnicity_source_concept_id']
    d = data_train.copy()
    for p_i in pop_fields:
        if p_i in d.columns:
            d.pop(p_i)
    one_hot_fields = [f for f in one_hot_fields if f in d.columns]
    d = pd.get_dummies(d, columns=one_hot_fields)
    return d


def model_static_patient_train(data_train, labels_train, config):

    d = model_static_patient_preprocess(data_train["person"])
    config["model"].fit(d, labels_train["death"])
    return config


def model_static_patient_predict(data_test, model):
    d = model_static_patient_preprocess(data_test["person"])
    return pd.DataFrame(model.predict_proba(d)[:, 1], index=data_test["person"]["person_id"], columns=["score"])


def apply_indexing(data, config, key="id"):
    index = model_configs.get_default_index(key=key)
    uids_iter = []
    uids_record = []
    for table_key, col_list in index.items():
        if "filter path" in config and os.path.exists(config["filter path"] + table_key + "_concepts.csv"):
            filter = pd.read_csv(config["filter path"] + table_key + "_concepts.csv")
        else:
            filter = {}
        for col_key in col_list:
            if isinstance(col_key, str):
                if col_key in filter:
                    filter_set = set(filter[col_key])
                    i_iter = [(col_key, i) for i in set(data[table_key][col_key].dropna()) if i in filter_set]
                    print(str((table_key, col_key, len(i_iter))))
                else:
                    i_iter = [(col_key, i) for i in set(data[table_key][col_key].dropna())]
                    print(str((table_key, col_key, len(i_iter))))
                if col_key == config["join field"]:
                    uids_record.extend(i_iter)
                else:
                    uids_iter.extend(i_iter)
            else:
                (col_key, (fn_field, join_field, fn)) = col_key
                data[table_key][fn_field] = data[table_key][fn_field].apply(fn)
                uids_iter.extend([(col_key, i) for i in set(data[table_key][col_key].dropna())])
                if join_field is not None:
                    uids_iter.extend([(join_field, (int(i), int(j))) for i,j in data[table_key][[join_field, fn_field]].dropna().drop_duplicates().values])
        print(str(table_key))
    uids_iter = set(uids_iter)
    uids_record = set(uids_record)
    return data, dict(zip(uids_record, range(len(uids_record)))), dict(zip(uids_iter, range(len(uids_iter)))), index


def get_grouped_features(data_train, config, uids_feats=None, key="id"):

    items = c.defaultdict(set)
    t = time.time()
    data_train, uids_records, uids_feats_ret, index = apply_indexing(data_train, config, key=key)
    if uids_feats is None:
        uids_feats = uids_feats_ret
    print("Built index: " + str(time.time()-t))

    join_field = config["join field"]

    t = time.time()
    pid_index = c.defaultdict(set)
    for table_key, col_list in index.items():
        for row in data_train[table_key].itertuples():
            row = dict(row._asdict())
            for col_key in col_list:
                if isinstance(col_key, str):

                    k_record = (join_field, row[join_field])
                    k_feat = (col_key, row[col_key] )
                    if col_key != join_field and k_feat in uids_feats and k_record in uids_records:
                        if len(pid_index):
                            for p_year in pid_index[row[join_field]]:
                                k_record = (join_field, (row[join_field], p_year))
                                items[uids_records[k_record]].add(uids_feats[k_feat])
                        else:
                            items[uids_records[k_record]].add(uids_feats[k_feat])


                else:
                    (col_key_iter, (fn_field, join_field_iter, fn)) = col_key
                    k_record = (join_field_iter, (row[join_field_iter], row[fn_field]))
                    k_feat = (col_key_iter, row[col_key_iter])

                    if k_feat in uids_feats and k_record in uids_records:
                        #print((k_record, k_feat))
                        pid_index[row[join_field_iter]].add(row[fn_field])
                        items[uids_records[k_record]].add(uids_feats[k_feat])
        print(str(table_key))
    print("Processed Index: " +str(time.time() - t))
    return items, uids_feats, uids_records

#get_sparse_person_features_mat(person_items, uids_records, labels, config, key="id", label_values=None, date_lag=(0))

def model_sparse_feature_test(data, config, uids,split_key="id", date_lag=[0]):
    t = time.time()
    p_ids = {j:i for i, j in enumerate(data["person"]["person_id"].copy())}

    person_items, uids_feats, uids_records = get_grouped_features(data, config, uids_feats=uids, key=split_key)

    if split_key == "dates":
        if "date lag" in config:
            date_lag = config["date lag"]
        data_sp, labels_translated = get_sparse_person_features_mat(person_items, uids_records, p_ids, config, key=split_key, date_lag=date_lag)
        p_ids_translated = list(labels_translated.keys())
        if isinstance(config["model"], model_configs.NNC):
            p = config["model"].predict_proba(data_sp.astype(np.float32))
        else:
            p = config["model"].predict_proba(data_sp)
        p, new_pids = get_grouped_preds(p, p_ids_translated, uids_records, p_ids=list(p_ids.keys()), date_lag=date_lag)
        keys_iter = pd.Series(new_pids, name="person_id")

    elif split_key=="id":
        print("Entering sparse processing")
        data_sp, labels_iter = get_sparse_person_features_mat(person_items, uids_records, p_ids, config, key=split_key)
        print(config["model"])
        #inference
        if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
            p, _ = evaluate_paired_model(config, data_sp)
        else:
            p = config["model"].predict_proba(data_sp)
        print("Finished inference")
        #p = config["model"].predict_proba(data_sp)
        keys_iter = pd.Series(list(p_ids.keys()), name="person_id")

    data = None
    print("Inference time:" + str(time.time() - t))
    p[p < 0] = 0
    p[p>1] = 1
    p[np.isnan(p)] = 0
    return pd.DataFrame(p[:, 1], index=keys_iter, columns=["score"])

def get_grouped_preds(p, keys_iter, uids_records,p_ids=None, date_lag=[0]):
    uids_records_rev = {v: k for k, v in uids_records.items()}
    pid_index = c.defaultdict(dict)
    for p_i, k_iter in zip(p, keys_iter):
        _, (p_key, p_year) = uids_records_rev[k_iter]
        pid_index[p_key][p_year] = p_i[1]
    p = []
    if p_ids is None:
        p_ids = list(pid_index.keys())

    for k in p_ids:
        if k not in pid_index:
            p.append([0, 0])
        else:
            d_iter = pid_index[k]
            pivot = max(list(d_iter.keys())) - max(date_lag)
            alive = [vv for kk, vv in d_iter.items() if kk < pivot]
            dead = [vv for kk, vv in d_iter.items() if kk >= pivot]
            if len(alive):
                v_a = max(dead) - max(alive)
            else:
                v_a = max(dead)
            p.append([0, v_a if v_a > 0 else 0])
    return np.array(p), p_ids

def preprocess_data(data, configs, uids=None, split_key="id"):
    if split_key=="id":
        labels_individual = {k:v for k,v in zip(data["death"]["person_id"].copy(), data["death"]["label"].copy())}

    else:
        labels_individual = {k: v for k, v in zip(data["death"]["person_id"].copy(), data["death"]["death_date"].copy())}
    config_base = list(configs.values())[0]

    date_lags = config_base["date lags"]

    person_items, uids_feats, uids_records = get_grouped_features(data, config_base, uids_feats=uids, key=split_key)
    data_sp, labels_iter = get_sparse_person_features_mat(person_items, uids_records, labels_individual, config_base,
                                                          key=split_key, date_lag=date_lags[0])

    config_base = model_configs.get_default_train_test(config_base)
    x_train, x_test, y_train, y_test, keys_train, keys_test = sk.model_selection.train_test_split(data_sp, list(labels_iter.values()), list(labels_iter.keys()), train_size=config_base["train size"])
    return x_train, x_test, y_train, y_test, keys_train, keys_test

def model_sparse_feature_cv_train(data, configs, uids=None, split_key="id"):
    t = time.time()
    if split_key=="id":
        labels_individual = {k:v for k,v in zip(data["death"]["person_id"].copy(), data["death"]["label"].copy())}

    else:
        labels_individual = {k: v for k, v in zip(data["death"]["person_id"].copy(), data["death"]["death_date"].copy())}
    labels_back = {k:v for k,v in zip(data["death"]["person_id"].copy(), data["death"]["label"].copy())}
    config_base = list(configs.values())[0]

    date_lags = config_base["date lags"]
    do_cv = config_base["do cv"]
    iters = config_base["cv iters"]

    person_items, uids_feats, uids_records = get_grouped_features(data, config_base, uids_feats=uids, key=split_key)
    data = None
    metrics_out = c.defaultdict(list)
    print("Data build time: " + str(time.time() - t))
    tt = time.time()
    if not do_cv:
        data_sp, labels_iter = get_sparse_person_features_mat(person_items, uids_records, labels_individual,
                                                           config_base, key=split_key)
        config_select = config_base
        if isinstance(config_select["model"], model_configs.NNC) or isinstance(config_select["model"], GridSearchCV):
            config_select["model"].param_grid["module__input_size"] = [data_sp.shape[1]]

            config_select["model"].fit(data_sp, np.array(list(labels_iter.values()), dtype=np.int64))
        elif isinstance(config_select["model"], dict) and "nets" in config_select["model"] and "pred" in config_select["model"]:
            train_paired_model(config_select, data_sp, np.array(list(labels_iter.values())))
        else:
            config_select["model"].fit(data_sp, np.array(list(labels_iter.values())))
        selected = "no CV"
        config_select["date lag"] = date_lags[0]
        perf = {}
    else:
        labels_store = {}
        for date_lag in date_lags:
            data_sp, labels_iter = get_sparse_person_features_mat(person_items, uids_records, labels_individual, config_base, key=split_key,  date_lag=date_lag)
            labels_store[tuple(date_lag)] = labels_iter

            for ii in range(iters):
                config_base = model_configs.get_default_train_test(config_base)
                x_train, x_test, y_train, y_test, keys_train, keys_test = sk.model_selection.train_test_split(data_sp, list(labels_iter.values()), list(labels_iter.keys()), train_size=config_base["train size"])
                print("CV Train data: " + str((x_train.shape, x_train.nnz, x_train.dtype)))
                print("CV Test data: " + str((x_test.shape, x_test.nnz, x_test.dtype)))
                #print(keys_train)
                for key_c, config in configs.items():
                    reset_net_model(config)
                x_apps_train = []
                x_apps_val = []
                for jj, (key_c, config) in enumerate(configs.items()):
                    key_c = (key_c, tuple(date_lag))
                    print("(Model,iteration, %): " + str((key_c, ii, jj/len(configs))))

                    #train
                    ttt = time.time()
                    if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
                        x_apps_train = train_paired_model(config, x_train, np.array(y_train), x_apps_train)
                    elif isinstance(config["model"], model_configs.NNC):
                        train_nn(config["model"], x_train, np.array(y_train))
                    else:
                        config["model"].fit(x_train, np.array(y_train))
                    print("CV Train: " + str(time.time() - ttt))

                    #eval
                    ttt = time.time()
                    if isinstance(config["model"], dict) and "nets" in config["model"] and "pred" in config["model"]:
                        y_pred, x_apps_val = evaluate_paired_model(config, x_test, x_apps_val)
                    elif (isinstance(config["model"], model_configs.PairedKnn) or isinstance(config["model"], model_configs.PairedPipeline)) and isinstance(config["model"].f_rep, model_configs.NNC) and hasattr(config["model"].f_rep, "module_"):
                        if config["model"].f_rep.module_.cache is not None:
                            inds_iter = config["model"].f_rep.module_.cache
                        else:
                            inds_iter = None
                        if config["model"].f_rep.module_.cache_rep is not None:
                            x_rep_iter = config["model"].f_rep.module_.cache_rep
                        else:
                            x_rep_iter = None
                        y_pred = config["model"].predict_proba(x_test, inds=inds_iter, x_test_t=x_rep_iter)
                    else:
                        y_pred = config["model"].predict_proba(x_test)
                    print("CV Predict: " + str(time.time() - ttt) )
                    if split_key == "dates":
                        y_pred, p_ids = get_grouped_preds(y_pred, keys_test, uids_records, p_ids=None, date_lag=date_lag)
                        y_test = [int(labels_back[k]) for k in p_ids]
                    print("CV Metrics")
                    metrics_out[key_c].append(sk.metrics.roc_auc_score(y_test, y_pred[:, 1]))
            perf = {k: {"mean": np.mean(v), "std": np.std(v)} for k,v in metrics_out.items()}
            selected, selected_mean = sorted({k:v["mean"] for k,v in perf.items()}.items(), key=operator.itemgetter(1))[::-1][0]
            config_select = configs[selected[0]]
            config_select["date lag"] = selected[1]
            print("Selected: " + str(selected))
            print(perf)
            print("Training full selected model")
            reset_net_model(config_select)
            if isinstance(config_select["model"], dict) and "nets" in config_select["model"] and "pred" in config_select["model"]:
                train_paired_model(config_select, data_sp, np.array(list(labels_store[selected[1]].values())))
            elif isinstance(config_select["model"], model_configs.NNC):
                train_nn(config_select["model"], data_sp, np.array(list(labels_store[selected[1]].values())))
            else:
                config_select["model"].fit(data_sp, list(labels_store[selected[1]].values()))

    config_select["train shape"] = data_sp.shape
    print("Model cv time: " + str(time.time() - tt))
    return config_select, selected, perf, metrics_out, configs, uids_feats
def evaluate_paired_model(config_select, data_sp, y_label=None, train=False, x_apps=[]):
    for k,vm in config_select["model"]["nets"].items():
        train_nn(vm, data_sp, y_label)
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

def train_nn(vm, data_sp, y_label):
    if (not hasattr(vm, "module_")) or vm.module_.training:
        s = data_sp.shape[1]
        vm.module__input_size = s
        vm.fit(data_sp, y_label)

def train_paired_model(config_select, data_sp, y_label, x_apps=[]):
    return evaluate_paired_model(config_select, data_sp, y_label=y_label, train=True, x_apps=x_apps)

def reset_net_model(config_select):
    if isinstance(config_select["model"], dict) and "nets" in config_select["model"]:
        for k, vm in config_select["model"]["nets"].items():
            vm.reset()
    elif isinstance(config_select["model"], model_configs.NNC):
        config_select["model"].reset()
    elif isinstance(config_select["model"], model_configs.PairedKnn) or isinstance(config_select["model"], model_configs.PairedPipeline):
        config_select["model"].reset()

def empirical_risk(person_feats, labels, th = 20):
    d = c.defaultdict(lambda: [0,0])
    for k,v in person_feats.items():
        for vv in list(v):
            d[vv] = list(map(operator.add, d[vv], [labels[k], 1]))
    d = {k:v[0]/v[1] if v[1]>th else 0 for k,v in d.items()}
    return d

def build_feature_graph(visit_feats, person_feats, th=20):
    d = c.defaultdict(int)
    for k, v in visit_feats.items():
        for p in it.combinations(v, 2):
            d[p]+=1
    d = {k:v for k, v in d.items() if v > th}
    values = list(d.values()) + list(d.values())
    k1 = [v[0] for v in d.keys()] + [v[1] for v in d.keys()]
    k2 = [v[1] for v in d.keys()]+[v[0] for v in d.keys()]
    d2 = sp.csr_matrix((values, (k1,k2)))
    return d

def get_dict_to_sparse(d1, shape=None, dtype=np.float32):
    a = []
    for k, v in enumerate(d1):
        a.extend([[k, vv] for vv in v])
    a = np.array(a)
    #print(a)
    if shape is None:
        return sp.csr_matrix((np.ones(a.shape[0]), (a[:, 0], a[:, 1])), dtype=dtype)
    else:
        return sp.csr_matrix((np.ones(a.shape[0]), (a[:, 0], a[:, 1])), shape=shape, dtype=dtype)

def get_sparse_person_features_mat(person_items, uids_records, labels_individual, config, key="id", date_lag=[0]):
    if key == "id":
        person_translated = [person_items[uids_records[("person_id", k)]] for k in labels_individual.keys()]
        labels_translated = labels_individual
    elif key == "dates":
        uids_records_rev = {v:k for k,v in uids_records.items()}
        label_index = set()
        for lag_i in list(date_lag):
            label_index = label_index.union(set([(i,j-lag_i) for i,j in labels_individual.items() if not np.isnan(j)]))
        labels_translated = {i:0 if uids_records_rev[i][1] not in label_index else 1 for i in person_items.keys()}
        person_translated = list(person_items.values())

    if "train shape" in config:
        person_sparse = get_dict_to_sparse(person_translated, shape=(len(person_translated), config["train shape"][1]))
    else:
        person_sparse = get_dict_to_sparse(person_translated)
    return person_sparse, labels_translated


def get_default_join(config, key="visits"):
    if key not in config or config[key] is None:
        if key == "visits":
            return "visit_occurrence_id"
        elif key == "person":
            return "person_id"
    else:
        return config[key]


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

    del globals()[var]