import hnswlib
import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB #ComplementNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import os
import numpy as np
#from autosklearn import classification as ask
import shutil
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import skorch
import joblib as jl
import app.models as model_includes
from pathlib import Path
import catboost as ct
os.environ["OMP_NUM_THREADS"] = "8"


class NNC(NeuralNetClassifier):
    def __init__(self, module, *args, **kwargs):
        super(NNC, self).__init__(module, *args, **kwargs)
    def train_nn(self, data_sp, y_label):
        if (not hasattr(self, "module_")) or self.module_.training:
            print("Fit: NNC")
            s = data_sp.shape[1]
            self.module__input_size = s
            self.fit(data_sp, y_label)
    def predict_proba(self, X):
        return super(NNC,self).predict_proba(X)
    def reset(self):
        if hasattr(self, "module_"):
            delattr(self, "module_")
            self.x_test_p = None
class DeepEHR(nn.Module):
    def __init__(self, input_size=6369, num_units1=100, num_units2=50,num_units3=25, num_units4=0, output_size=2, nonlinear=F.relu, dropout=0.5):
        super(DeepEHR, self).__init__()
        self.sizes = [input_size, num_units1, num_units2, num_units3, num_units4]
        self.dense0 = nn.Linear(input_size, num_units1)
        self.nonlinear = nonlinear
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units1, num_units2)
        self.dense2 = nn.Linear(num_units2, num_units3)
        self.train_preds =None
        self.cache = None
        self.x_train_rep = None
        if num_units3 and num_units4:
            self.dense3 = nn.Linear(num_units3, num_units4)
        else:
            self.dense3 = None
        self.output = nn.Linear([i for i in self.sizes if i != 0][-1], output_size)

    def forward(self, X, **kwargs):
        X = self.nonlinear(self.dense0(X))
        #X = self.dropout(X)
        X = F.relu(self.dense1(X))
        #X = self.dropout(X)
        X = F.relu(self.dense2(X))
        if self.dense3 is not None:
            X = self.dropout(X)
            X = F.relu(self.dense3(X))
        if "transform" in kwargs:
            return X
        else:
            return F.softmax(self.output(X), dim=-1)

class LDA_classifier:
    def __init__(self, **kwargs):
        self.model = sk.decomposition.LatentDirichletAllocation(**kwargs)
        self.weights = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.weights = self.get_factor_risk_vector(X, y)

    def predict_proba(self, X):
        return self.evaluate_factor_risk(X)

    def get_factor_risk_vector(self, x, y):
        y_int = [-1 if not x else 1 for x in y]
        x_tran = self.model.transform(x)
        aa = np.sum(x_tran * np.repeat(np.expand_dims(np.array(y_int), 1), x_tran.shape[1], axis=1), axis=0)
        return 1 - (aa - max(aa)) / (abs(max(aa)) - abs(min(aa)))

    def evaluate_factor_risk(self, x):
        x_tran = self.model.transform(x)
        risks = np.sum(np.repeat(np.expand_dims(self.weights, 0), x_tran.shape[0], axis=0) * x_tran, axis=1)
        return np.vstack((np.zeros(len(risks)), risks)).transpose()


class PairedKnn:

    def __init__(self, f_rep, f_pred, n_max=20, metric="cosine", use_labels=False, ef=64, M =64):
        self.f_rep = f_rep
        self.f_pred = f_pred
        self.x_train_p = None
        self.n_max = n_max
        self.metric = metric
        self.ef = ef
        self.M = M
        self.use_labels=use_labels

    def reset(self):
        reset_model(self.f_rep)
        reset_model(self.f_pred)
        self.x_train_p = None

    def fit(self, x_train, y_train):

        if isinstance(self.f_rep, NNC) and (not hasattr(self.f_rep, "module_") or self.f_rep.module_.x_train_rep is None):
            print("Fit Rep NN: PairedKNN")
            self.f_rep.train_nn(x_train, y_train)
            self.f_rep.module_.x_train_rep = self.f_rep.infer(skorch.utils.to_tensor(x_train, device="cpu", accept_sparse=True), transform=True).data.numpy()

            #self.f_rep.module_.knn_index.fit(self.f_rep.module_.x_train_rep)
            #self.f_rep.module_.knn_index = sk.neighbors.NearestNeighbors(n_jobs=-1, leaf_size=self.leaf_size, n_neighbors=self.n_max,
            #                                               metric=self.metric)
            self.f_rep.module_.knn_index = hnswlib.Index(space=self.metric, dim=self.f_rep.module_.x_train_rep.shape[1])
            self.f_rep.module_.knn_index.init_index(max_elements=self.f_rep.module_.x_train_rep.shape[0], ef_construction=self.ef, M=self.M)
            self.f_rep.module_.knn_index.set_num_threads(-1)
            self.f_rep.module_.knn_index.add_items(self.f_rep.module_.x_train_rep)
            print("Create KNN Index: PairedKNN")

        if isinstance(self.f_pred, NNC):
            self.f_pred.train_nn(x_train, y_train)
        else:
            if not is_fitted(self.f_pred, x_train):
                print("Fit Pred Other: PairedKNN")
                self.f_pred.fit(x_train, y_train)
        if self.x_train_p is None:
            if self.use_labels:
                self.x_train_p = np.column_stack([np.logical_not(y_train), y_train])
            else:
                self.x_train_p = self.f_pred.predict_proba(x_train)

    def predict_proba(self, x_test, inds=None):
        if isinstance(self.f_rep, NNC):
            x_test_t = self.f_rep.infer(skorch.utils.to_tensor(x_test, device="cpu", accept_sparse=True),transform=True).data.numpy()
            print("Computed Test Representation: PairedKNN")

        if inds is None or (self.f_rep.module_.cache is not None and self.n_max > self.f_rep.module_.cache.shape[1]):
            #dists, inds = self.f_rep.module_.knn_index.kneighbors(x_test_t)
            inds, dists = self.f_rep.module_.knn_index.knn_query(x_test_t, k=self.n_max)
            print("Computed KNN inds: PairedKNN")
        if isinstance(self.f_rep, NNC):
            if (self.f_rep.module_.cache is not None and inds.shape[1] > self.f_rep.module_.cache.shape[1]) or (self.f_rep.module_.cache is None):
                self.f_rep.module_.cache = inds
        r1 = []
        r2 = []
        print("Predicting: KNN")
        for vi in inds:
            r1.append(np.mean(self.x_train_p[vi[0:self.n_max]][:, 0]))
            r2.append(np.mean(self.x_train_p[vi[0:self.n_max]][:, 1]))
        ret = np.column_stack((r1, r2))
        return ret


class PairedPipeline:
    def __init__(self, f_rep, f_pred):
        self.f_rep = f_rep
        self.f_pred = f_pred


    def reset(self):
        reset_model(self.f_rep)
        reset_model(self.f_pred)


    def fit(self, x_train, y_train):
        if isinstance(self.f_rep, NNC) and (not hasattr(self.f_rep, "module_") or self.f_rep.module_.x_train_rep is None):
            print("Fit Rep NN: PairedPipeline")
            self.f_rep.train_nn(x_train, y_train)
            self.f_rep.module_.x_train_rep = self.f_rep.infer(skorch.utils.to_tensor(x_train, device="cpu", accept_sparse=True), transform=True).data.numpy()
        if isinstance(self.f_pred, NNC):
            self.f_pred.train_nn(self.f_rep.module_.x_train_rep, y_train)
        else:
            if not is_fitted(self.f_pred, self.f_rep.module_.x_train_rep):
                print("Fit pred: PairedPipeline")
                self.f_pred.fit(self.f_rep.module_.x_train_rep, y_train)
            else:
                print("Skipped fit pred: PairedPipeline")
    def predict_proba(self, x_test, inds=None):
        if isinstance(self.f_rep, NNC):
            x_test_t = self.f_rep.infer(skorch.utils.to_tensor(x_test, device="cpu", accept_sparse=True),transform=True).data.numpy()
            print("Computed Representation: Pipeline")
        else:
            x_test_t= self.f_rep(x_test)
        print("Predicting: Pipeline")
        return self.f_pred.predict_proba(x_test_t)

class LargeEnsemble:
    def __init__(self, models, keys=None, ensemble=GradientBoostingClassifier, ensemble_params={}):
        self.models = models
        self.keys = keys
        self.ensemble = ensemble(**ensemble_params)
    def fit(self, x_train, y_train):
        for m in self.models:
            if isinstance(m, NNC):
                m.train_nn(x_train, y_train)
            elif isinstance(m, PairedPipeline) or isinstance(m, PairedKnn):
                m.fit(x_train, y_train)
            else:
                if not is_fitted(m, x_train):
                    print("Ensemble train:" + str(m))
                    m.fit(x_train, y_train)
        ps = self._get_preds(x_train)
        self.ensemble.fit(ps, y_train)

    def _get_preds(self, x):
        ps = []
        for m in self.models:
            ps.append(m.predict_proba(x)[:, 1])
        return np.stack(ps, axis=1)

    def reset(self):
        for m in self.models:
            reset_model(m)
        reset_model(self.ensemble)
    def predict_proba(self, x_test):
        ps = self._get_preds(x_test)
        return self.ensemble.predict_proba(ps)

def dummy_ret(x):
    return x

def reset_model(m):
    if isinstance(m, dict) and "nets" in m:
        for k, vm in m["nets"].items():
            reset_model(vm)
    if isinstance(m, dict) and "preds" in m:
        reset_model(m["preds"])
    elif isinstance(m, (NNC, PairedKnn, PairedPipeline, LargeEnsemble)):
        m.reset()
    else:
        m.__init__(**m.get_params())

def is_fitted(m, data):
    try:
        m.predict_proba(data[0:1, :])
        return 1
    except (NotFittedError, xgb.core.XGBoostError):
        return 0

def pickle_nms(config, file):
    if isinstance(config["model"], PairedKnn) and isinstance(config["model"].f_rep, NNC) and hasattr(config["model"].f_rep, "module_") and hasattr(config["model"].f_rep.module_, "knn_index"):
        config["model"].f_rep.module_.knn_index.save_index(file + "_knn")
        delattr(config["model"].f_rep.module_, "knn_index")
    jl.dump(config, file)

def unpickle_nms(file):
    config = jl.load(file)
    if Path(file+"_knn").is_file():
        r = hnswlib.Index(space="cosine", dim=50)
        r.load_index(file+"_knn")
        config["model"].f_rep.module_.knn_index = r
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
    config["train path"] = "../train/"
    config["test path"] = "../infer/"
    config["model path"] = "../model/"
    config["output path"] = "../output/"
    config["scratch path"] = "../scratch/"
    config["filter path"] = "./filter/"
    config["train"] = True
    config["do cv"] = True
    config["date lags"] = [[0]]
    config["join field"] = "person_id"
    config["cv iters"] = 1
    config["cv split key"] = "id"
    return config


def get_rf_baseline_config(model_params={"max_depth": 100, "n_estimators":200,"n_jobs":-1}, name=None):
    config = get_base_config(RandomForestClassifier, model_params, name=name)
    return config


def get_naivebayes_baseline_config(model_params={}, name=None):
    config = get_base_config(BernoulliNB, model_params, name=name)
    return config


def get_xgboost_baseline_config(model_params={"max_depth":10, "n_jobs:":-1, "n_estimators":100}, name=None):
    return get_base_config(xgb.sklearn.XGBClassifier, model_params, name=name)


def get_baseline_cv_configs():
    configs = dict()
    # configs["aut10o"] = get_base_config(model_fn=ask.AutoSklearnClassifier, model_params={"time_left_for_this_task":1500, "per_run_time_limit":300,
    #                                              "n_jobs":8,
    #                                              "ensemble_size":3, "ensemble_nbest":20, "ml_memory_limit":30000})
    #
    # tmp_path = configs["auto"]["model path"] + "tmp"
    # if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
    #     shutil.rmtree(tmp_path)
    # out_path = configs["auto"]["model path"] + "out"
    # if os.path.exists(out_path) and os.path.isdir(out_path):
    #     shutil.rmtree(out_path)
    # configs["auto"]["model"].tmp_folder = configs["auto"]["model path"] + "tmp"
    # configs["auto"]["model"].output_folder = configs["auto"]["model path"] + "out"
    # #


    # configs["LDA-10"] = get_base_config(model_fn=LDA_classifier, model_params={"learning_method":"online", "batch_size":1000, "n_jobs":-1,
    #                                                                            "n_components":10})
    # configs["LDA-20"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 20})
    # configs["LDA-30"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 30})
    # configs["LDA-40"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #                                                   "n_components": 40})
    # configs["LDA-50"] = get_base_config(model_fn=LDA_classifier,
    #                                     model_params={"learning_method": "online", "batch_size": 1000, "n_jobs": -1,
    #
    #
    #                                                   "n_components": 50})

    # nnets = {}
    # m_epochs = 100
    # p3 = {
    #     'lr': 0.1,
    #     'batch_size': 1024,
    #     'module__dropout': 0,
    #     'module__num_units1': 100,
    #     'module__num_units2': 100,
    #     'module__num_units3': 50,
    #     'module__num_units4': 50,
    #     'max_epochs':m_epochs,
    #     'train_split': skorch.dataset.CVSplit(.4, stratified=True),
    #     'iterator_train__shuffle':True,
    #     'callbacks': [skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=5, threshold=0.0001, threshold_mode='rel',
    #                                lower_is_better=True)]}
    #
    # configs["3layer-50"] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **p3})
    # nnets["3layer-50"]=configs["3layer-50"]["model"]
    #
    # p3 = {
    #     'lr': 0.1,
    #     'batch_size': 1024,
    #     'module__dropout': 0,
    #     'module__num_units1': 500,
    #     'module__num_units2': 200,
    #     'module__num_units3': 100,
    #     'module__num_units4': 25,
    #     'max_epochs':m_epochs,
    #     'train_split': skorch.dataset.CVSplit(.4, stratified=True),
    #     'iterator_train__shuffle':True,
    #     'callbacks': [skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=5, threshold=0.0001, threshold_mode='rel',
    #                                lower_is_better=True)]}
    #
    # configs["3layer"] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **p3})
    # nnets["3layer"]=configs["3layer"]["model"]



    # p4 = {
    #     'lr': 0.1,
    #     'batch_size': 1024,
    #     'module__dropout':0,
    #     'module__num_units1': 10,
    #     'module__num_units2': 20,
    #     'module__num_units3': 10,
    #     'module__num_units4': 0,
    #     'max_epochs':m_epochs,
    #     'train_split': skorch.dataset.CVSplit(.3, stratified=True),
    #     'iterator_train__shuffle':True,
    #     'callbacks': [skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=5, threshold=0.0001, threshold_mode='rel',
    #                                lower_is_better=True)]}
    #
    # configs["3layer-small"] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **p4})
    # nnets["3layer-small"]=configs["3layer-small"]["model"]

    # for k in paired_ks:
    #     configs[("3layer-50",k)] = get_base_config(model_fn=PairedKnn, model_params={"f_rep":configs["3layer-50"]["model"], "f_pred":configs["3layer-50"]["model"], "n_max":k})
    #     configs[("3layer", k)] = get_base_config(model_fn=PairedKnn,
    #                                                   model_params={"f_rep": configs["3layer"]["model"],
    #                                                                 "f_pred": configs["3layer"]["model"],
    #                                                                 "n_max": k})

    # nnets = {}
    # p3 = {
    #     'lr': 0.1,
    #     'batch_size': 1024,
    #     'module__dropout': 0.5,
    #     'module__num_units1': 200,
    #     'module__num_units2': 50,
    #     'module__num_units3': 50,
    #     'module__num_units4': 25,
    #     'max_epochs':100,
    #     'train_split': skorch.dataset.CVSplit(.3, stratified=True),
    #     'iterator_train__shuffle':True,
    #     'callbacks': [skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=3, threshold=0.0001, threshold_mode='rel',
    #                                lower_is_better=True)]}
    #
    # configs["3layer-50-do"] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **p3})
    # nnets["3layer-50-do"]=configs["3layer-50-do"]["model"]
    # p4 = {
    #     'lr': 0.1,
    #     'batch_size':1024,
    #     'module__dropout': 0.2,
    #     'module__num_units1': 500,
    #     'module__num_units2': 100,
    #     'module__num_units3': 50,
    #     'module__num_units4': 0,
    #     'max_epochs':50,
    #     'module__nonlinear': F.leaky_relu,
    #     'iterator_train__shuffle':True,
    #     'train_split': skorch.dataset.CVSplit(.3, stratified=True),
    #     'callbacks': [skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=3, threshold=0.0001, threshold_mode='rel',
    #                                        lower_is_better=True)]
    # }
    #
    # configs["3layer-leaky"] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **p4})
    # nnets["3layer-leaky"]=configs["3layer-leaky"]["model"]


    # net = NNC(
    #     DeepEHR,
    #     max_epochs=10,
    #     lr=0.1,
    #     module__num_units=100,
    #     # Shuffle training data on each epoch
    #     iterator_train__shuffle=True,
    # )
    # configs["net100"] = net
    #here

    #configs["gb"] = get_base_config()
    #configs["knn-25"] = get_base_config(model_fn=KNeighborsClassifier,
    #                                                  model_params={"n_neighbors": 25})
    #configs["knn-50"] = get_base_config(model_fn=KNeighborsClassifier,model_params={"n_neighbors": 50, "n_jobs":-1 })
    #configs["knn-100"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 100})
    #configs["knn-200"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 200})
    #configs["knn-300"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 300})
    #configs["knn-500"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 500})
    #configs["knn-1000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 1000})
    #configs["knn-2000"] = get_base_config(model_fn=KNeighborsClassifier, model_params={"n_neighbors": 2000})
    # configs["nb"] = get_base_config(model_fn=BernoulliNB)
    #configs["nb-Compliment"] = get_base_config(model_fn=ComplementNB)
    #configs["nb-Multi"] = get_base_config(model_fn=MultinomialNB)
    #configs["random stratified"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "stratified"})
    #configs["random uniform"] = get_base_config(model_fn=DummyClassifier,model_params={"strategy": "uniform"})
    # configs["rf"] = get_rf_baseline_config()












###### XGB

    #
    #
    # model_xgbs = {}
    # p = {"max_depth": 12, "nthread":4, "eval_metric":"auc"}
    # objectives = ["binary:logistic"]
    # ns = [500]
    # sample_type = ["weighted"]
    # alphas = [0]
    # lambdas = [1]
    # feature_selector = []
    # maxes = [8]
    # boosters = ["gbtree"]
    # trees = ["hist"]
    # scale_pos_weights = [1, 10]
    # for o in objectives:
    #     for n in ns:
    #         for m in maxes:
    #             for b in boosters:
    #                     p2 = p.copy()
    #                     p2["n_estimators"] = n
    #                     p2["booster"] = b
    #                     p2["max_depth"] = m
    #                     p2["objective"] = o
    #                     if b == "gbtree":
    #                         for s in scale_pos_weights:
    #                             p2["scale_pos_weight"] = s
    #                             for a in alphas:
    #                                 p2["alpha"] = a
    #                                 for l in lambdas:
    #                                     p2["lambda"] = l
    #                                     if l != a:
    #                                         for t1 in trees:
    #                                             p2["tree_method"] = t1
    #                                             mm = get_xgboost_baseline_config(model_params=p2.copy())
    #                                             #mm["model"] = {"nets":nnets, "pred":mm["model"]}
    #                                             configs[("xgboost",n, o,b, t1, s, a, l)] = mm
    #                                             model_xgbs[("xgboost",n, o,b, t1, s, a, l)] = mm
    #
    #                     elif b == "dart":
    #                         for st in sample_type:
    #                             p2["sample_type"] = st
    #                             mm = get_xgboost_baseline_config(model_params=p2.copy())
    #                             configs[("xgboost", n, o, b, st)] = mm
    #                             model_xgbs[("xgboost", n, o, b, st)] = mm
    #                     elif b == "gblinear":
    #                         for fs in feature_selector:
    #                             p2["feature_selector"] = fs
    #                             p2["updater"] = "coord_descent"
    #                             for a in alphas:
    #                                 p2["alpha"] = a
    #                                 for l in lambdas:
    #                                     p2["lambda"] = l
    #                                     if l != a:
    #                                         mm = get_xgboost_baseline_config(model_params=p2.copy())
    #                                         ##mm["model"] = {"nets": nnets, "pred":mm["model"]}
    #                                         configs[("xgboost",n, o, b, fs, a, l)] = mm
    #                                         model_xgbs[("xgboost",n, o, b, fs, a, l)] = mm
    #                     else:
    #                         configs[("xgboost",n, o, b)] = get_xgboost_baseline_config(
    #                             model_params=p2)

    #p = {"max_depth": 12, "nthread":4, "eval_metric":"auc"}
    depths = [5]
    objectives = ["Logloss"]
    ns = [1000]
    lrs = [0.01,0.05, 0.1]
    l2_leaf_regs = [1, 9, 18]
    rsms = [0.5, 1]
    class_weights = [(.1, .9), (0.25, 0.75)]

    for d in depths:
        for o in objectives:
            for lr in lrs:
                for l2 in l2_leaf_regs:
                    for rsm in rsms:
                        for w in class_weights:
                            for n in ns:
                                configs[("catboost", d, o, lr, l2, rsm, w, n)] = get_base_config(model_fn=ct.CatBoostClassifier, model_params={"logging_level":"Silent", "n_estimators":n, "depth":d,"loss_function": o, "learning_rate":lr, "l2_leaf_reg":l2, "rsm":rsm, "class_weights":w})


    #
    # sample_type = ["weighted"]
    # alphas = [0, 1]
    # lambdas = [1]
    # feature_selector = ["cyclic", "shuffle"]
    # maxes = [8]
    # boosters = ["gbtree", "gblinear"]
    # trees = ["auto", "hist"]
    # scale_pos_weights = [1, 10, 20]

    # paired_ks = [50, 25, 20, 10]
    # #f_rep, f_pred, n_max = 20, metric = "cosine", use_labels = False, ef = 64, M = 64
    # for k in paired_ks:
    #     for ii, (kk, v_model) in enumerate(model_xgbs.items()):
    #         # v_iter = {"3layer-50": get_xgboost_baseline_config(model_params=v_model["model"].get_params())["model"],
    #         #           "3layer": get_xgboost_baseline_config(model_params=v_model["model"].get_params())["model"],
    #         #           "3layer-small": get_xgboost_baseline_config(model_params=v_model["model"].get_params())["model"]}
    #         configs[("3layer-50", kk, k, False)] = get_base_config(model_fn=PairedKnn,
    #                                                   model_params={"f_rep": configs["3layer-50"]["model"],
    #                                                                 "f_pred": v_model["model"],
    #                                                                 "n_max": k,
    #                                                                 "use_labels":False})
    #         configs[("3layer", kk, k, False)] = get_base_config(model_fn=PairedKnn,
    #                                                   model_params={"f_rep": configs["3layer"]["model"],
    #                                                                 "f_pred": v_model["model"],
    #                                                                 "n_max": k,
    #                                                                 "use_labels":False})
    #         if ii == 0:
    #             configs[("3layer-50", kk, k, True)] = get_base_config(model_fn=PairedKnn,
    #                                                                         model_params={
    #                                                                             "f_rep": configs["3layer-50"]["model"],
    #                                                                             "f_pred": v_model["model"],
    #                                                                             "n_max": k,
    #                                                                             "use_labels": True})
    #             configs[("3layer", kk, k, True)] = get_base_config(model_fn=PairedKnn,
    #                                                                      model_params={
    #                                                                          "f_rep": configs["3layer"]["model"],
    #                                                                          "f_pred": v_model["model"],
    #                                                                          "n_max": k,
    #                                                                          "use_labels": True})
        # configs[("3layer-50", kk, "pipeline")] = get_base_config(model_fn=PairedPipeline,
        #                                           model_params={"f_rep": configs["3layer-50"]["model"],
        #                                                         "f_pred": v_iter["3layer-50"]})
        # configs[("3layer", kk, "pipeline")] = get_base_config(model_fn=PairedPipeline,
        #                                           model_params={"f_rep": configs["3layer"]["model"],
        #                                                        "f_pred": v_iter["3layer"]})
        # configs[("3layer-small", kk, "pipeline")] = get_base_config(model_fn=PairedPipeline,
        #                                           model_params={"f_rep": configs["3layer-small"]["model"],
        #                                                        "f_pred": v_iter["3layer-small"]})
    # configs_add = {}
    # ensemble_iters = 3
    # ensemble_choice= [3, 6]
    # for i in range(ensemble_iters):
    #     for j in ensemble_choice:
    #         a_iter = list(configs.items())
    #         items_iter = [a_iter[i] for i in np.random.choice(len(configs), size=j, replace=False)]
    #         m_iter = [v["model"] for k,v in items_iter]
    #         k_iter = [k for k,v in items_iter]
    #         configs_add[("ensemble", i, j)] = get_base_config(model_fn=LargeEnsemble, model_params={"models": m_iter, "keys":k_iter, "ensemble":sk.linear_model.LogisticRegressionCV, "ensemble_params":{}})
    #
    # configs = dict(list(configs.items())+list(configs_add.items()))
    print("Models #: " + str(len(configs)), flush=True)
    return configs


def str_to_year(x):
    return int(x[0:4]) if isinstance(x,str) else np.nan


def get_default_index(key="id"):
    if key=="id":
        return {"condition_occurrence": ["condition_concept_id"],
                "procedure_occurrence": ["procedure_concept_id"],
                "observation":["observation_concept_id"],
                "measurement": ["measurement_concept_id"],
                "drug_exposure": ["drug_concept_id"],
                "person": ["year_of_birth", "gender_concept_id", "race_concept_id", "person_id"]}
    elif key=="dates":
        return {"condition_occurrence": [("condition_concept_id", ["condition_start_date","person_id",str_to_year ])],
                "procedure_occurrence": [("procedure_concept_id", ["procedure_date", "person_id", str_to_year])],
                "measurement": [("measurement_concept_id", ["measurement_date", "person_id", str_to_year])],
                "drug_exposure": [("drug_concept_id", ["drug_exposure_start_date", "person_id", str_to_year])],
                "person": ["year_of_birth", "gender_concept_id", "race_concept_id", "person_id"]}
def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config
