import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import os
from sklearn.exceptions import NotFittedError
import numpy as np
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import skorch
import joblib as jl
from pathlib import Path
import catboost as ct
import torch
os.environ["OMP_NUM_THREADS"] = "8"

class NNC(NeuralNetClassifier):
    def __init__(self, module, *args, **kwargs):
        super(NNC, self).__init__(module, *args, **kwargs)
    def train_nn(self, data_sp, y_label):
        if (not hasattr(self, "module_")) or self.module_.training:
            print("Fit: NNC")
            s = data_sp.shape[1]
            self.module__input_size = s
            self.fit(data_sp, torch.from_numpy(y_label).long())
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

            self.f_rep.module_.knn_index = NearestNeighbors(metric=self.metric, n_jobs=-1)
            self.f_rep.module_.knn_index.fit(self.f_rep.module_.x_train_rep)
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
            dists, inds = self.f_rep.module_.knn_index.kneighbors(x_test_t, n_neighbors=self.n_max)
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
    elif isinstance(m, AdaBoostClassifier):
        m = sk.base.clone(m)
    else:
        m.__init__(**m.get_params())

def is_fitted(m, data):
    try:
        m.predict_proba(data[0:1, :])
        return 1
    except (NotFittedError, xgb.core.XGBoostError):
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


def get_rf_baseline_config(model_params={"max_depth": 100, "n_estimators":200,"n_jobs":-1}, name=None):
    config = get_base_config(RandomForestClassifier, model_params, name=name)
    return config


def get_naivebayes_baseline_config(model_params={}, name=None):
    config = get_base_config(BernoulliNB, model_params, name=name)
    return config


def get_xgboost_baseline_config(model_params={"max_depth":10, "n_jobs:":-1, "n_estimators":100}, name=None):
    return get_base_config(xgb.sklearn.XGBClassifier, model_params, name=name)

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
    nets = {}
    if "embed" in model_names:
            nets = get_net_params()
            configs = {**configs, **nets}

    if "embed-knn" in model_names:
        paired_ks = [3,5,7]
        if not len(nets):
            nets = get_net_params()
        for net_k, net in nets.items():
            for k in paired_ks:
                configs[(net_k,k)] = get_base_config(model_fn=PairedKnn, model_params={"f_rep":nets[net_k]["model"], "f_pred":nets[net_k]["model"], "n_max":k})

    print("Models #: " + str(len(configs)))
    return configs


def build_net_params(lr=0.05, batch_size=1024, dropout=0.2,units= [200, 100, 50, 10], nonlinear=F.relu, max_epochs=1000,
                     train_splits=skorch.dataset.CVSplit(.3, stratified=True),
                     callbacks=[skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=10, threshold=0.00001, threshold_mode='rel',
                                               lower_is_better=True)], shuffle=True):
    return {'lr': lr,
            'batch_size': batch_size,
            'module__dropout': dropout,
            'module__num_units1': units[0],
            'module__num_units2': units[1],
            'module__num_units3': units[2],
            'module__num_units4': units[3],
            'max_epochs': max_epochs,
            'train_split': train_splits,
            'module__nonlinear': nonlinear,
            'iterator_train__shuffle': shuffle,
            'callbacks': callbacks}

def get_net_params(configs={}):
    lrs = [0.05]
    ps = dict()

    for lr in lrs:
        ps[("4-layer", lr)] = build_net_params(units=[100, 50, 25, 10], lr=lr)
        ps[("3-layer", lr)] = build_net_params(units=[100, 50, 10, 0], lr=lr)
        ps[("4-layer-small", lr)] = build_net_params(units=[20, 20, 10, 10], lr=lr)
        ps[("3-layer-small", lr)] = build_net_params(units=[20, 10, 10, 0], lr=lr)
        ps[("4-layer-small-leaky", lr)] = build_net_params(units=[20, 20, 10, 10], nonlinear=F.leaky_relu, lr=lr)
        ps[("3-layer-small-leaky", lr)] = build_net_params(units=[20, 10, 10, 0], nonlinear=F.leaky_relu, lr=lr)
        for k,v in ps.items():
            configs[k] = get_base_config(model_fn=NNC, model_params={"module": DeepEHR, **v})
    return configs


def get_default_train_test(config):
    if "train size" not in config:
        config["train size"] = 0.5
    return config