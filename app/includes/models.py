import pandas as pd
import numpy as np
import pathlib
import csv
import collections as c
import os
import sys
import operator
import sklearn as sk

import scipy.sparse as sp


os.chdir(os.path.dirname(sys.argv[0]))

def read_ehrdc_data(path, data_keys=None, label_keys=None, useful_keys=None, expand_labels=("person", "person_id"),
                    filter_useful=True):
    if data_keys is None:
        data_keys = ["condition_occurrence", "drug_exposure", "measurement", "observation_period", "observation",
                     "person", "procedure_occurrence", "visit_occurrence"]
    if label_keys is None:
        label_keys = ["death"]

    print(os.getcwd())
    with open("includes/OMOP_useful_columns.csv") as f:
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
            data[k] = pd.read_csv(f, usecols=useful_keys[k])
            data[k] = data[k].dropna(axis=1, how="all")
    for k in label_keys:
        f = path + k + ".csv"
        if pathlib.Path(f).exists():
            labels[k] = pd.read_csv(path + k + ".csv", usecols=useful_keys[k])
            if expand_labels:
                positives = set(labels[k][expand_labels[1]])
                inds = data[expand_labels[0]][expand_labels[1]]
                data[k] = pd.concat([inds, inds.apply(lambda x, label: x in label, args=[positives])], axis=1,
                                    keys=[inds.name, "label"])
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
    model = get_default_model(config)
    d = model_static_patient_preprocess(data_train["person"])
    model.fit(d, labels_train["death"])
    return model


def model_static_patient_predict(data_test, model):
    d = model_static_patient_preprocess(data_test["person"])
    return pd.DataFrame(model.predict_proba(d)[:, 1], index=data_test["person"]["person_id"], columns=["score"])


def get_grouped_features(data_train, config):
    graph_modalities = get_default_graph_modalities(config)

    visits_items = c.defaultdict(set)
    person_items = c.defaultdict(set)
    uids = dict()
    uid_iter= 0

    joins = [(k, get_default_join(config, key=k)) for k in ["person", "visits"]]

    for j_name, j_key in joins:
        for table_key, col_key in graph_modalities:
            if j_key in data_train[table_key]:
                for row in data_train[table_key].itertuples():
                    row = row._asdict()
                    if not np.isnan(row[j_key]):
                        v_rev = (j_key, row[j_key])
                        c_rev = (col_key, row[col_key])
                        # p_rev = (person_col, row[person_col])
                        if v_rev not in uids:
                            uids[v_rev] = uid_iter
                            uid_iter += 1
                        if c_rev not in uids:
                            uids[c_rev] = uid_iter
                            uid_iter += 1
                        if j_name == "visits":
                            visits_items[uids[v_rev]].add(uids[c_rev])
                        if j_name == "person" and col_key != j_key:
                            person_items[uids[v_rev]].add(uids[c_rev])
    return visits_items, person_items, uids

# def do_model_selection(data, iters=10, keep_models=True):
#     configs = {}
#     for depth in [4, 8, 16]:
#         configs[("max_depth", depth)] =  get_rf_baseline_config()


def model_sparse_feature_cv(data, configs, iters=10, data_sp=None, uids=None):

    p_ids = data["death"]["person_id"]
    labels = data["death"]["label"]
    config_base = list(configs.values())[0]
    if data_sp is None or uids is None:
        data_sp, uids = get_sparse_person_features_mat(data, p_ids, config_base)
    metrics_out = c.defaultdict(list)
    for ii in range(iters):
        config_base = get_default_train_test(config_base)
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data_sp, labels, train_size=config_base["train size"])
        for key_c, config in configs.items():
            print("(Model, iteration): " + str((key_c, ii)))
            config["model"].fit(x_train, y_train)
            y_pred = config["model"].predict_proba(x_test)[:, 1]
            metrics_out[key_c].append(sk.metrics.roc_auc_score(y_test, y_pred))
    perf = {k: {"mean": np.mean(v), "std": np.std(v)} for k,v in metrics_out.items()}
    selected, selected_mean = sorted({k:v["mean"] for k,v in perf.items()}.items(), key=operator.itemgetter(1))[::-1][0]

    model = sk.base.clone(configs[selected]["model"])
    model.fit(data_sp, labels)
    return model, selected, perf, metrics_out, configs

def get_dict_to_sparse(d1):
    a = []
    for k, v in d1.items():
        a.extend([[k, vv] for vv in v])
    a = np.array(a)
    return sp.csr_matrix((np.ones(a.shape[0]), (a[:, 0], a[:, 1])))


def get_sparse_person_features_mat(data, labels, config):
    _, person_items, uids = get_grouped_features(data, config)
    person_translated = {i: person_items[uids[("person_id", k)]] for i, k in enumerate(labels)}
    person_sparse = get_dict_to_sparse(person_translated)
    return person_sparse, uids


def get_default_join(config, key="visits"):
    if key not in config or config[key] is None:
        if key == "visits":
            return "visit_occurrence_id"
        elif key == "person":
            return "person_id"
    else:
        return config[key]


def get_default_graph_modalities(config, key="graph modalities"):
    if key not in config or config[key] is None:
        return (("person", "person_id"), ("person", "gender_concept_id"), ("person", "race_concept_id"),
                ("condition_occurrence", "condition_type_concept_id"), ("procedure_occurrence", "procedure_concept_id"),
                ("drug_exposure", "drug_concept_id"))
    else:
        return config[key]


def get_default_model(config):
    if "model" not in config or config["model"] is None:
        return sk.naive_bayes.BernoulliNB()
    else:
        return config["model"]

def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config


# def get_jaccard_knn(d1, d2, kn=50):
#     rets = {}
#     for k,v in d1.items():
#         r = {k2:jaccard_dist(v, v2) for k2,v2 in d2.items() if k != k2}
#         rs = sorted(r.items(), key=operator.itemgetter(1))
#         rets[k] = [k for k,v in rs]
#         print(k)
# def jaccard_dist(a, b):
#     return len(a.intersection(b))/len(a.union(b))

