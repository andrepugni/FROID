import numpy as np
import pandas as pd
import copy
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.base import BaseEstimator, ClassifierMixin
from src.classes.feature_enrich import dim_reduction_class, outlier_detection_class
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import  RandomOverSampler
import time
# from cure import Cure
import random
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel


class FROID(BaseEstimator, ClassifierMixin):
    def __init__(self, clf_base, method, list_methods_od=[], list_methods_red=[], seed=42, meta='FROID',
                 filename='stars', write_train=False, write_test=False, scaling=True, scaler=RobustScaler,
                 feature_sel='nofs', expanded=False, previous_stage='OD-BINOD-RED', store=False):
        self.seed = seed
        self.clf_base = copy.deepcopy(clf_base)
        self.clf_err = None
        self.method = method
        self.list_methods_od = list_methods_od
        self.list_methods_red = list_methods_red
        self.method_list = method.split("^")
        self.dict_method = {k: {} for k in range(len(self.method_list))}
        self.filename = filename
        self.scores = None
        self.time_to_expand = None
        self.method_list_od_upd = copy.deepcopy(list_methods_od)
        self.method_list_red_upd = copy.deepcopy(list_methods_red)
        self.write_train = write_train
        self.write_test = write_test
        self.meta = meta
        self.scaling = scaling
        self.scaler = scaler()
        self.feature_sel = feature_sel
        self.expanded = expanded
        self.previous_stage = previous_stage
        self.store = store

    def fit(self, X, y, sample_weight=None):

        X1 = X.copy()
        original_columns = X1.columns.tolist().copy()
        X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
        X1.dropna(axis=1, how='any', inplace=True)
        k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
                      100, 150, 200, 250]
        k_list = [k for k in k_list_pre if k < X1.shape[0]]
        kl_list_pre = [1, 5, 10, 20]
        kl_list = [k for k in kl_list_pre if k < X1.shape[0]]
        self.to_rem_list = []
        self.kl_list = kl_list
        self.nu_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        self.n_list = [10, 20, 50, 70, 100, 150, 200, 250]
        self.k_list = k_list
        self.dict_atts = {0: X1.columns.tolist()}
        counts = np.unique(y, return_counts=True)
        self.perc_train = np.min(counts[1]) / len(y)
        atts_ = X1.columns.tolist()
        all_atts_ = X1.columns.tolist()
        time_start = time.time()
        self.atts_ = atts_
        self.pre_fs_atts_ = atts_
        if self.expanded:
            orig_feats = [col for col in atts_ if ('_ORIG_' not in col)]
            self.dict_atts[-1] = orig_feats.copy()
            inf_mean_cols = X1.columns[np.where((np.mean(X1) == np.inf)|(np.mean(X1) == -np.inf))].tolist()
            inf_var_cols = X1.columns[np.where((np.var(X1) == np.inf)|(np.var(X1) == -np.inf))].tolist()
            self.dict_atts[0] = [col for col in atts_ if ('_ORIG_' in col) and
                                 (col not in inf_var_cols) and (col not in inf_mean_cols)]
        for i, el in enumerate(self.method_list):
            # print(el)
            level_cols = []
            if (i == 0) & (self.expanded is False):
                colname = 'ORIG'
            elif self.expanded is True:
                colname = self.previous_stage
            else:
                colname = '{}{}'.format(el, i - 1)
            # Dimensionality Reduction Expansion
            if el == 'RED':
                for j, string_method in enumerate(self.list_methods_red):
                    print("starting REDUCTION METHODS")
                    if j//10 == 0:
                        print(string_method)
                    m = dim_reduction_class(X1[self.dict_atts[i]], y, string_method)
                    self.dict_method[i][string_method] = m
                    if m is not None:
                        # scores_train
                        if string_method in ['TSNE', 'MDS', 'SE']:
                            res_train = m.fit_transform(X1[self.dict_atts[i]])
                        else:
                            res_train = m.transform(X1[self.dict_atts[i]])
                        cols = ["{}_comp_{}_{}".format(string_method, colname, comp) for comp in
                                range(m.n_components)]
                        level_cols += cols
                        atts_ += cols
                        tmp = pd.DataFrame(res_train, index=X1.index, columns=cols)
                        if j == 0:
                            print("there we go")
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        continue
            # Outlier Detection Expansion
            elif el in ['OD', 'BINOD', 'OD-BINOD']:
                print(el)
                for j, string_method in enumerate(self.list_methods_od):
                    print(string_method)
                    if 'DBSCAN' in string_method:
                        cl, m = outlier_detection_class(X1[self.dict_atts[i]], string_method,
                                                        perc_train=self.perc_train)
                        try:
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            self.dict_method[i][string_method] = cl, m
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            atts_ += cols_to_add
                            all_atts_ += cols
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
                        for k_num, k in enumerate(self.k_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_K{}".format(k)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            # print(cols[0])
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            else:
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['SKL_ISO_N']:
                        for k_num, n in enumerate(self.n_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, n=n,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_N{}".format(n)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_N{}".format(colname, string_method, n),
                                    "bin_scores_{}_{}_N{}".format(colname, string_method, n)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['PYOD_OCSVM_NU']:
                        for k_num, nu in enumerate(self.nu_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, nu=nu,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_NU{}".format(nu)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_NU{}".format(colname, string_method, nu),
                                    "bin_scores_{}_{}_NU{}".format(colname, string_method, nu)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method == 'LOP_basic':
                        for k_num, k in enumerate(self.kl_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_K{}".format(k)] = m
                            scores_X = m.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif 'SKL_ELLENV' in string_method:
                        m = outlier_detection_class(X1[self.dict_atts[i]], string_method, perc_train=self.perc_train)
                        self.dict_method[i][string_method] = m
                        bin_scores_X = m.predict(X1[self.dict_atts[i]])
                        bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                        scores_X = m.decision_function(X1[self.dict_atts[i]])
                        maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                        cols = ["scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method)]
                        if el == 'OD':
                            cols_to_add = [col for col in cols if 'bin_' not in col]
                        elif el == 'BINOD':
                            cols_to_add = [col for col in cols if 'bin_' in col]
                        else:
                            cols_to_add = cols
                        level_cols += cols_to_add
                        all_atts_ += cols
                        atts_ += cols_to_add
                        tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                                           index=X.index)
                        tmp = tmp.astype(float)
                        if j == 0:
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    else:

                        m = outlier_detection_class(X1[self.dict_atts[i]], string_method, perc_train=self.perc_train)
                        self.dict_method[i][string_method] = m
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            self.to_rem_list += cols
            # if method use both od and feature reduction
            elif el in ['OD-RED', 'BINOD-RED', 'OD-BINOD-RED']:
                print(el)
                for j, string_method in enumerate(self.list_methods_od):
                    # print(string_method)
                    # DBSCAN based
                    if 'DBSCAN' in string_method:
                        cl, m = outlier_detection_class(X1[self.dict_atts[i]], string_method,
                                                        perc_train=self.perc_train)
                        try:
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            self.dict_method[i][string_method] = cl, m
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            if el == 'OD-RED':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD-RED':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            atts_ += cols_to_add
                            all_atts_ += cols
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    # KNN based methods
                    elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
                        for k_num, k in enumerate(self.k_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_K{}".format(k)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            # print(cols[0])
                            if el == 'OD-RED':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD-RED':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    # Isolation Forest methods
                    elif string_method in ['SKL_ISO_N']:
                        for k_num, n in enumerate(self.n_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, n=n,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_N{}".format(n)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_N{}".format(colname, string_method, n),
                                    "bin_scores_{}_{}_N{}".format(colname, string_method, n)]
                            if el == 'OD-RED':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD-RED':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    # SVM based methods
                    elif string_method in ['PYOD_OCSVM_NU']:
                        for k_num, nu in enumerate(self.nu_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, nu=nu,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_NU{}".format(nu)] = m
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_NU{}".format(colname, string_method, nu),
                                    "bin_scores_{}_{}_NU{}".format(colname, string_method, nu)]
                            if el == 'OD-RED':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD-RED':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    # LOOP methods
                    elif string_method == 'LOP_basic':
                        for k_num, k in enumerate(self.kl_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            self.dict_method[i][string_method + "_K{}".format(k)] = m
                            scores_X = m.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            if el == 'OD':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            else:
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    # Elliptic Envelope method
                    elif 'SKL_ELLENV' in string_method:
                        m = outlier_detection_class(X1[self.dict_atts[i]], string_method, perc_train=self.perc_train)
                        self.dict_method[i][string_method] = m
                        bin_scores_X = m.predict(X1[self.dict_atts[i]])
                        bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                        scores_X = m.decision_function(X1[self.dict_atts[i]])
                        maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                        cols = ["scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method)]
                        if el == 'OD-RED':
                            cols_to_add = [col for col in cols if 'bin_' not in col]
                        elif el == 'BINOD-RED':
                            cols_to_add = [col for col in cols if 'bin_' in col]
                        else:
                            cols_to_add = cols
                        level_cols += cols_to_add
                        all_atts_ += cols
                        atts_ += cols_to_add
                        tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                                           index=X.index)
                        tmp = tmp.astype(float)
                        if j == 0:
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    # other methods
                    else:
                        m = outlier_detection_class(X1[self.dict_atts[i]], string_method, perc_train=self.perc_train)
                        self.dict_method[i][string_method] = m
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            if el == 'OD-RED':
                                cols_to_add = [col for col in cols if 'bin_' not in col]
                            elif el == 'BINOD-RED':
                                cols_to_add = [col for col in cols if 'bin_' in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            self.to_rem_list += cols
                for j, string_method in enumerate(self.list_methods_red):
                    m = dim_reduction_class(X1[self.dict_atts[i]], y, string_method)
                    self.dict_method[i][string_method] = m
                    if m is not None:
                        # scores_train
                        if string_method in ['TSNE', 'MDS', 'SE']:
                            res_train = m.fit_transform(X1[self.dict_atts[i]])
                        else:
                            res_train = m.transform(X1[self.dict_atts[i]])
                        cols = ["{}_comp_{}_{}".format(string_method, colname, comp) for comp in
                                range(m.n_components)]
                        level_cols += cols
                        atts_ += cols
                        tmp = pd.DataFrame(res_train, index=X1.index, columns=cols)
                        X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        continue
            if el != 'ORIG':
                X1 = X2.copy()
            try:
                print("so far so good pt 1")
                print(X2.head())
            except:
                print("so far except pt 1")
                print(X1.head())

            self.dict_atts[i + 1] = [raw for raw in list(np.array(level_cols).flatten())
                                     if raw not in self.to_rem_list]
            self.atts_ = [el for el in atts_ if el not in self.to_rem_list]
            self.all_atts_ = [el for el in atts_]
            X2 = None
            if (self.expanded == True) & (el == 'ORIG') & (self.dict_atts[0] != []):
                tmp = {0: self.dict_atts[-1].copy(), 1: self.dict_atts[0].copy()}
                self.dict_atts = tmp
            elif (self.expanded == True) & (el == 'ORIG') & (self.dict_atts[0] == []):
                tmp = {0: self.dict_atts[-1].copy(), 1: self.dict_atts[0].copy()}
                self.dict_atts = tmp
            if self.feature_sel in ['variance', 'selectmodel']:
                self.scaling = True
            if self.scaling:
                print("here we are going to scale with X1")
                print(X1.head())
                self.pre_fs_atts_ = []
                X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
                null_cols = X1.columns[X1.isnull().any()].tolist().copy()
                self.to_rem_list += null_cols.copy()
                if i == 0:
                    for k, v in self.dict_atts.items():
                        self.pre_fs_atts_ += v
                else:
                    for k, v in self.dict_atts.items():
                        if v in self.to_rem_list:
                            continue
                        else:
                            self.pre_fs_atts_ += v

                self.scaling_feats = [el for el in self.pre_fs_atts_ if (el not in null_cols) &
                                      (el not in original_columns)]
                self.scaler.fit(X1[self.scaling_feats])
                X1[self.scaling_feats] = self.scaler.transform(X1[self.scaling_feats])
                print("here we have scaled with X1")
                print(X1.head())
                X1 = X1.copy()
            if (self.feature_sel == 'variance'):
                fs = VarianceThreshold(threshold=.2)
                self.fs = fs
                # try:
                self.pre_fs_atts_ = self.scaling_feats.copy()
                X1[self.pre_fs_atts_] = X1[self.pre_fs_atts_].astype(np.float64).round(10)
                try:
                    print(max(X1.max()))
                    self.fs.fit(X1[self.pre_fs_atts_])
                except ValueError:
                    X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
                    null_cols = X1.columns[X1.isnull().any()].tolist().copy()
                    self.pre_fs_atts_ = [col for col in self.pre_fs_atts_ if col not in null_cols]
                    self.fs.fit(X1[self.pre_fs_atts_])
                    # import pdb; pdb.set_trace()
                    # X1[self.pre_fs_atts_] = X1[self.pre_fs_atts_].astype(np.float64).round(4)
                    # self.fs.fit(X1[self.pre_fs_atts_])
                X2 = pd.DataFrame(self.fs.transform(X1[self.pre_fs_atts_]),
                                  columns=self.fs.get_feature_names_out().tolist())
                to_rem = [el for el in X1.columns.tolist() if el not in X2.columns.tolist()]
                self.to_rem_list += to_rem
                X1 = X2.copy()
                X2 = None
            elif (self.feature_sel == 'selectmodel'):
                fs = SelectFromModel(self.clf_base)
                self.pre_fs_atts_ = self.scaling_feats.copy()
                self.fs = fs
                self.fs.fit(X1[self.pre_fs_atts_], y)
                X2 = pd.DataFrame(self.fs.transform(X1[self.pre_fs_atts_]),
                                  columns=self.fs.get_feature_names_out().tolist())
                to_rem = [el for el in X1.columns.tolist() if el not in X2.columns.tolist()]
                self.to_rem_list += to_rem
                X1 = X2.copy()
                X2 = None
            else:
                self.fs = None
            self.dict_atts[i + 1] = [raw for raw in list(np.array(level_cols).flatten())
                                     if raw not in self.to_rem_list]
            self.atts_ = [el for el in atts_ if el not in self.to_rem_list]
            self.all_atts_ = [el for el in atts_ if el not in self.to_rem_list]
        time_end = time.time()
        self.time_to_expand = time_end - time_start
        # db_test = pd.DataFrame(res_test, index=X.index, columns=cols)
        if self.write_train:
            if self.expanded:
                path_dataset = 'data/{}/{}_{}_{}^{}_{}_train.csv'.format(self.filename, self.filename,
                                                                         self.meta, self.previous_stage,
                                                                         self.method,
                                                                         self.feature_sel)
            else:
                path_dataset = 'data/{}/{}_{}_{}_{}_train.csv'.format(self.filename, self.filename,
                                                                      self.meta, self.method,
                                                                      self.feature_sel)
            pd.concat([X1, y], axis=1).to_csv(path_dataset, index=False)

        if self.store:
            self.data_train = X1[self.atts_].copy()
        self.clf_base.fit(X1[self.atts_], y, sample_weight)

    def predict_proba(self, X):
        X1 = X.copy()
        X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
        X1.dropna()
        for i, el in enumerate(self.method_list):
            # print(el)
            if (i == 0) & (self.expanded is False):
                colname = 'ORIG'
            elif self.expanded is True:
                colname = self.previous_stage
            else:
                colname = '{}{}'.format(el, i - 1)
            if el == 'ORIG':
                pass
            elif el == 'RED':
                level_cols = []
                for j, string_method in enumerate(self.list_methods_red):
                    # print(string_method)
                    m = copy.deepcopy(self.dict_method[i][string_method])
                    # scores_train
                    if m is not None:
                        if string_method in ['TSNE', 'MDS', 'SE']:
                            res_test = m.fit_transform(X1[self.dict_atts[i]])
                        else:
                            res_test = m.transform(X1[self.dict_atts[i]])
                        cols = ["{}_comp_{}_{}".format(string_method, colname, comp) for comp in
                                range(m.n_components)]
                        level_cols += cols
                        tmp = pd.DataFrame(res_test, index=X1.index, columns=cols)
                        if j == 0:
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        continue
            elif el in ['OD', 'BINOD', 'OD-BINOD']:
                level_cols = []
                for j, string_method in enumerate(self.list_methods_od):
                    # print(string_method)
                    if 'DBSCAN' in string_method:
                        cl, m = outlier_detection_class(X1[self.dict_atts[i]], string_method,
                                                        perc_train=self.perc_train)
                        try:
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
                        for k_num, k in enumerate(self.k_list):
                            m = self.dict_method[i][string_method + "_K{}".format(k)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['SKL_ISO_N']:
                        for k_num, n in enumerate(self.n_list):
                            m = self.dict_method[i][string_method + "_N{}".format(n)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_N{}".format(colname, string_method, n),
                                    "bin_scores_{}_{}_N{}".format(colname, string_method, n)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['PYOD_OCSVM_NU']:
                        for k_num, nu in enumerate(self.nu_list):
                            m = self.dict_method[i][string_method + "_NU{}".format(nu)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_NU{}".format(colname, string_method, nu),
                                    "bin_scores_{}_{}_NU{}".format(colname, string_method, nu)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method == 'LOP_basic':
                        for k_num, k in enumerate(self.kl_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            scores_X = m.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif 'SKL_ELLENV' in string_method:
                        m = self.dict_method[i][string_method]
                        bin_scores_X = m.predict(X1[self.dict_atts[i]])
                        bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                        scores_X = m.decision_function(X1[self.dict_atts[i]])
                        maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                        cols = ["scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method)]
                        tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                                           index=X1.index)
                        tmp = tmp.astype(float)
                        if j == 0:
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        m = self.dict_method[i][string_method]
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            continue
            elif el in ['OD-RED', 'BINOD-RED', 'OD-BINOD-RED']:
                level_cols = []
                for j, string_method in enumerate(self.list_methods_od):
                    # print(string_method)
                    if 'DBSCAN' in string_method:
                        cl, m = outlier_detection_class(X1[self.dict_atts[i]], string_method,
                                                        perc_train=self.perc_train)
                        try:
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
                        for k_num, k in enumerate(self.k_list):
                            m = self.dict_method[i][string_method + "_K{}".format(k)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['SKL_ISO_N']:
                        for k_num, n in enumerate(self.n_list):
                            m = self.dict_method[i][string_method + "_N{}".format(n)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_N{}".format(colname, string_method, n),
                                    "bin_scores_{}_{}_N{}".format(colname, string_method, n)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method in ['PYOD_OCSVM_NU']:
                        for k_num, nu in enumerate(self.nu_list):
                            m = self.dict_method[i][string_method + "_NU{}".format(nu)]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}_NU{}".format(colname, string_method, nu),
                                    "bin_scores_{}_{}_NU{}".format(colname, string_method, nu)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif string_method == 'LOP_basic':
                        for k_num, k in enumerate(self.kl_list):
                            m = outlier_detection_class(X1[self.dict_atts[i]], string_method, k=k,
                                                        perc_train=self.perc_train)
                            scores_X = m.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = ["scores_{}_{}_K{}".format(colname, string_method, k),
                                    "bin_scores_{}_{}_K{}".format(colname, string_method, k)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if (j == 0) & (k_num == 0):
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                    elif 'SKL_ELLENV' in string_method:
                        m = self.dict_method[i][string_method]
                        bin_scores_X = m.predict(X1[self.dict_atts[i]])
                        bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                        scores_X = m.decision_function(X1[self.dict_atts[i]])
                        maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                        cols = ["scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method)]
                        tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                                           index=X1.index)
                        tmp = tmp.astype(float)
                        if j == 0:
                            X2 = pd.concat([X1, tmp], axis=1)
                        else:
                            X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        m = self.dict_method[i][string_method]
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = ["scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method)]
                            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=X1.index, columns=cols)
                            tmp = tmp.astype(float)
                            if j == 0:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            continue
                for j, string_method in enumerate(self.list_methods_red):
                    # print(string_method)
                    m = copy.deepcopy(self.dict_method[i][string_method])
                    # scores_train
                    if m is not None:
                        if string_method in ['TSNE', 'MDS', 'SE']:
                            res_test = m.fit_transform(X1[self.dict_atts[i]])
                        else:
                            res_test = m.transform(X1[self.dict_atts[i]])
                        cols = ["{}_comp_{}_{}".format(string_method, colname, comp) for comp in
                                range(m.n_components)]
                        level_cols += cols
                        tmp = pd.DataFrame(res_test, index=X1.index, columns=cols)
                        X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        continue
            if el not in ['ORIG']:
                X1 = X2.copy()
                X2 = None
        if (self.scaling == True):
            X1[self.scaling_feats] = self.scaler.transform(X1[self.scaling_feats])
            X1 = X1.copy()
            if self.feature_sel == 'variance':
                if self.fs is not None:
                    X2 = pd.DataFrame(self.fs.transform(X1[self.pre_fs_atts_]),
                                      columns=self.fs.get_feature_names_out().tolist())
                    X1 = X2.copy()
                    X2 = None
            elif self.feature_sel == 'selectmodel':
                X2 = pd.DataFrame(self.fs.transform(X1[self.pre_fs_atts_]),
                                  columns=self.fs.get_feature_names_out().tolist())
                X1 = X2.copy()
                X2 = None
            else:
                self.fs = None
                # db_test = pd.DataFrame(res_test, index=X.index, columns=cols)
        if self.write_test:
            if self.expanded:
                path_dataset = 'data/{}/{}_{}_{}^{}_{}_test.csv'.format(self.filename, self.filename,
                                                                        self.meta, self.previous_stage,
                                                                        self.method,
                                                                        self.feature_sel)
            else:
                path_dataset = 'data/{}/{}_{}_{}_{}_test.csv'.format(self.filename, self.filename,
                                                                     self.meta, self.method,
                                                                     self.feature_sel)
            X1.to_csv(path_dataset, index=False)
        if self.store:
            self.data_test = X1.copy()
        try:
            self.scores = self.clf_base.predict_proba(X1[self.atts_])
        except:
            import pdb;
            pdb.set_trace()
        return self.scores

    def predict(self, X):
        if self.scores is not None:
            return np.argmax(self.scores, axis=1)
        else:
            return np.argmax(self.predict_proba(X), axis=1)