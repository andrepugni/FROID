import numpy as np
import pandas as pd
import copy

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from src.classes.cure import Cure
from src.classes.feature_enrich import dim_reduction_class, outlier_detection_class
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import time

# from cure import Cure
import random
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel


class FROID(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method,
        list_methods_od=[],
        list_methods_red=[],
        seed=42,
        meta="FROID",
        store=True,
        expanded=False,
        inception=False,
        previous_stage="OD-BINOD-RED",
    ):
        self.method = method
        self.seed = seed
        self.list_methods_od = list_methods_od
        self.list_methods_red = list_methods_red
        self.method_list = method.split("^")
        self.dict_method = {k: {} for k in range(len(self.method_list))}
        self.scores = None
        self.time_to_expand = 0
        self.method_list_od_upd = copy.deepcopy(list_methods_od)
        self.method_list_red_upd = copy.deepcopy(list_methods_red)
        self.meta = meta
        self.expanded = expanded
        self.previous_stage = previous_stage
        self.store = store
        self.inception = inception

    def fit(self, X, y, scaler="robust"):
        """

        :param X:
        :param y:
        :return:
        """
        if scaler == "robust":
            self.scaler = RobustScaler()
        X1 = X.copy()
        original_columns = X1.columns.tolist().copy()
        # here we drop all inf and nan columns
        X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
        X1.dropna(axis=1, how="any", inplace=True)
        k_list_pre = [
            1,
            2,
            3,
            4,
            5,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            150,
            200,
            250,
        ]
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
        X2 = None
        # if expanded it means the columns have already some features coming from FROID
        if self.expanded:
            orig_feats = [
                col
                for col in atts_
                if ("_ORIG_" not in col)
                and ("_RED" not in col)
                and ("_OD-BINOD" not in col)
            ]
            self.dict_atts[-1] = orig_feats.copy()
            # here we drop columns for which both mean and variance is exploded
            inf_mean_cols = X1.columns[
                np.where((np.mean(X1) == np.inf) | (np.mean(X1) == -np.inf))
            ].tolist()
            inf_var_cols = X1.columns[
                np.where((np.var(X1) == np.inf) | (np.var(X1) == -np.inf))
            ].tolist()
            self.dict_atts[0] = [
                col
                for col in atts_
                if ("_ORIG_" in col)
                and (col not in inf_var_cols)
                and (col not in inf_mean_cols)
            ]
        for i, el in enumerate(self.method_list):
            # print(el)
            level_cols = []
            if (i == 0) & (self.expanded is False) & (self.inception is False):
                colname = "ORIG"
            elif self.inception:
                colname = self.previous_stage
            else:
                colname = "{}{}".format(el, i - 1)
            # Dimensionality Reduction Expansion
            if el == "RED":
                for j, string_method in enumerate(self.list_methods_red):
                    if j == 0:
                        print("starting REDUCTION METHODS")
                    if j // 10 == 0:
                        print(string_method)
                    try:
                        m = dim_reduction_class(X1[self.dict_atts[i]], y, string_method)
                        self.dict_method[i][string_method] = m
                        # here we try to fit the method over the data, if it fails we do not use it for Dimensionality Reduction
                        if m is not None:
                            # scores_train
                            if string_method in ["TSNE", "MDS", "SE"]:
                                res_train = m.fit_transform(X1[self.dict_atts[i]])
                            else:
                                res_train = m.transform(X1[self.dict_atts[i]])
                            cols = [
                                "{}_comp_{}_{}".format(string_method, colname, comp)
                                for comp in range(m.n_components)
                            ]
                            level_cols += cols
                            atts_ += cols
                            tmp = pd.DataFrame(res_train, index=X1.index, columns=cols)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            continue
                    except:
                        continue
            # Outlier Detection Expansion
            elif el in ["OD", "BINOD", "OD-BINOD"]:
                print(el)
                for j, string_method in enumerate(self.list_methods_od):
                    if j // 10 == 0:
                        print(string_method)
                    if "DBSCAN" in string_method:
                        cl, m = outlier_detection_class(
                            X1[self.dict_atts[i]],
                            string_method,
                            perc_train=self.perc_train,
                        )
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
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            if el == "OD":
                                cols_to_add = [col for col in cols if "bin_" not in col]
                            elif el == "BINOD":
                                cols_to_add = [col for col in cols if "bin_" in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            atts_ += cols_to_add
                            all_atts_ += cols
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in [
                        "PYOD_KNN_Largest",
                        "PYOD_KNN_Median",
                        "PYOD_KNN_Mean",
                        "SKL_LOF",
                    ]:
                        for k_num, k in enumerate(self.k_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                # print(cols[0])
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                else:
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["SKL_ISO_N"]:
                        for k_num, n in enumerate(self.n_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    n=n,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_N{}".format(n)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                    "bin_scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                ]
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["PYOD_OCSVM_NU"]:
                        for k_num, nu in enumerate(self.nu_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    nu=nu,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_NU{}".format(nu)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                    "bin_scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                ]
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method == "LOP_basic":
                        for k_num, k in enumerate(self.kl_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ] = m
                                scores_X = m.local_outlier_probabilities
                                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif "SKL_ELLENV" in string_method:
                        m = outlier_detection_class(
                            X1[self.dict_atts[i]],
                            string_method,
                            perc_train=self.perc_train,
                        )
                        self.dict_method[i][string_method] = m
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method),
                            ]
                            if el == "OD":
                                cols_to_add = [col for col in cols if "bin_" not in col]
                            elif el == "BINOD":
                                cols_to_add = [col for col in cols if "bin_" in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X, maha_scores_X],
                                columns=cols,
                                index=X.index,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            self.to_rem_list += cols
                    else:
                        try:
                            m = outlier_detection_class(
                                X1[self.dict_atts[i]],
                                string_method,
                                perc_train=self.perc_train,
                            )
                            self.dict_method[i][string_method] = m
                            if m is not None:
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method),
                                ]
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            else:
                                cols = [
                                    "scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method),
                                ]
                                self.to_rem_list += cols
                        except:
                            continue
            # if method use both od and feature reduction
            elif el in ["OD-RED", "BINOD-RED", "OD-BINOD-RED"]:
                print(el)
                for j, string_method in enumerate(self.list_methods_od):
                    print(string_method)
                    # DBSCAN based
                    if "DBSCAN" in string_method:
                        cl, m = outlier_detection_class(
                            X1[self.dict_atts[i]],
                            string_method,
                            perc_train=self.perc_train,
                        )
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
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            if el == "OD-RED":
                                cols_to_add = [col for col in cols if "bin_" not in col]
                            elif el == "BINOD-RED":
                                cols_to_add = [col for col in cols if "bin_" in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            atts_ += cols_to_add
                            all_atts_ += cols
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    # KNN based methods
                    elif string_method in [
                        "PYOD_KNN_Largest",
                        "PYOD_KNN_Median",
                        "PYOD_KNN_Mean",
                        "SKL_LOF",
                    ]:
                        for k_num, k in enumerate(self.k_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                # print(cols[0])
                                if el == "OD-RED":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD-RED":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    # Isolation Forest methods
                    elif string_method in ["SKL_ISO_N"]:
                        for k_num, n in enumerate(self.n_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    n=n,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_N{}".format(n)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                    "bin_scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                ]
                                if el == "OD-RED":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD-RED":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    # SVM based methods
                    elif string_method in ["PYOD_OCSVM_NU"]:
                        for k_num, nu in enumerate(self.nu_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    nu=nu,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_NU{}".format(nu)
                                ] = m
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                    "bin_scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                ]
                                if el == "OD-RED":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD-RED":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    # LOOP methods
                    elif string_method == "LOP_basic":
                        for k_num, k in enumerate(self.kl_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ] = m
                                scores_X = m.local_outlier_probabilities
                                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                if el == "OD":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                else:
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    # Elliptic Envelope method
                    elif "SKL_ELLENV" in string_method:
                        m = outlier_detection_class(
                            X1[self.dict_atts[i]],
                            string_method,
                            perc_train=self.perc_train,
                        )
                        self.dict_method[i][string_method] = m
                        if m is not None:
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method),
                            ]
                            if el == "OD-RED":
                                cols_to_add = [col for col in cols if "bin_" not in col]
                            elif el == "BINOD-RED":
                                cols_to_add = [col for col in cols if "bin_" in col]
                            else:
                                cols_to_add = cols
                            level_cols += cols_to_add
                            all_atts_ += cols
                            atts_ += cols_to_add
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X, maha_scores_X],
                                columns=cols,
                                index=X.index,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            self.to_rem_list += cols
                    # other methods
                    else:
                        try:
                            m = outlier_detection_class(
                                X1[self.dict_atts[i]],
                                string_method,
                                perc_train=self.perc_train,
                            )
                            self.dict_method[i][string_method] = m
                            if m is not None:
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method),
                                ]
                                if el == "OD-RED":
                                    cols_to_add = [
                                        col for col in cols if "bin_" not in col
                                    ]
                                elif el == "BINOD-RED":
                                    cols_to_add = [col for col in cols if "bin_" in col]
                                else:
                                    cols_to_add = cols
                                level_cols += cols_to_add
                                all_atts_ += cols
                                atts_ += cols_to_add
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            else:
                                cols = [
                                    "scores_{}_{}".format(colname, string_method),
                                    "bin_scores_{}_{}".format(colname, string_method),
                                ]
                                self.to_rem_list += cols
                        except:
                            continue
                for j, string_method in enumerate(self.list_methods_red):
                    try:
                        m = dim_reduction_class(X1[self.dict_atts[i]], y, string_method)
                        self.dict_method[i][string_method] = m
                        if m is not None:
                            # scores_train
                            if string_method in ["TSNE", "MDS", "SE"]:
                                res_train = m.fit_transform(X1[self.dict_atts[i]])
                            else:
                                res_train = m.transform(X1[self.dict_atts[i]])
                            cols = [
                                "{}_comp_{}_{}".format(string_method, colname, comp)
                                for comp in range(m.n_components)
                            ]
                            level_cols += cols
                            atts_ += cols
                            tmp = pd.DataFrame(res_train, index=X1.index, columns=cols)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            continue
                    except:
                        continue
            if el != "ORIG":
                X1 = X2.copy()
            try:
                print("so far so good pt 1")
                print(X2.head())
            except:
                print("so far except pt 1")
                print(X1.head())
            self.dict_atts[i + 1] = [
                raw
                for raw in list(np.array(level_cols).flatten())
                if raw not in self.to_rem_list
            ]
            self.atts_ = [el for el in atts_ if el not in self.to_rem_list]
            X2 = None
            if (self.expanded == True) & (el == "ORIG") & (self.dict_atts[0] != []):
                tmp = {0: self.dict_atts[-1].copy(), 1: self.dict_atts[0].copy()}
                self.dict_atts = tmp
            elif (self.expanded == True) & (el == "ORIG") & (self.dict_atts[0] == []):
                tmp = {0: self.dict_atts[-1].copy(), 1: self.dict_atts[0].copy()}
                self.dict_atts = tmp
            print("here we are going to scale with X1")
            print(X1.head(1))
            # self.pre_fs_atts_ = []
            X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
            # null_cols = X1.columns[X1.isnull().any()].tolist().copy()
            # bin_scores_cols = [col for col in X1.columns if "bin_scores_" in col]
            # self.to_rem_list += null_cols.copy()
            # if i == 0:
            #     for k, v in self.dict_atts.items():
            #         self.pre_fs_atts_ += v
            # else:
            #     for k, v in self.dict_atts.items():
            #         if v in self.to_rem_list:
            #             continue
            #         else:
            #             self.pre_fs_atts_ += v
            #
            # self.scaling_feats = [el for el in self.pre_fs_atts_ if (el not in null_cols) &
            #                       (el not in original_columns) & (el not in bin_scores_cols)]
            # print(self.scaling_feats)
            # self.scaler.fit(X1[self.scaling_feats])
            # X1[self.scaling_feats] = self.scaler.transform(X1[self.scaling_feats])
            # print("here we have scaled with X1")
            # print(X1.head(1))
            X1 = X1.copy()
            self.dict_atts[i + 1] = [
                raw
                for raw in list(np.array(level_cols).flatten())
                if raw not in self.to_rem_list
            ]
            self.atts_ = [el for el in atts_ if el not in self.to_rem_list]
        time_end = time.time()
        self.time_to_expand = time_end - time_start
        # db_test = pd.DataFrame(res_test, index=X.index, columns=cols)
        self.data_train = X1[self.atts_].copy()

    def transform(self, X):
        X1 = X.copy()
        X1 = X1.replace([np.inf, -np.inf], np.nan).copy()
        X1.dropna(inplace=True)
        X2 = None
        for i, el in enumerate(self.method_list):
            # print(el)
            if (i == 0) & (self.expanded is False) & (self.inception is False):
                colname = "ORIG"
            elif self.inception:
                colname = self.previous_stage
            else:
                colname = "{}{}".format(el, i - 1)
            if el == "ORIG":
                pass
            elif el == "RED":
                level_cols = []
                for j, string_method in enumerate(self.list_methods_red):
                    print(string_method)
                    try:
                        m = copy.deepcopy(self.dict_method[i][string_method])
                        # scores_train
                        if m is not None:
                            if string_method in ["TSNE", "MDS", "SE"]:
                                res_test = m.fit_transform(X1[self.dict_atts[i]])
                            else:
                                res_test = m.transform(X1[self.dict_atts[i]])
                            cols = [
                                "{}_comp_{}_{}".format(string_method, colname, comp)
                                for comp in range(m.n_components)
                            ]
                            level_cols += cols
                            tmp = pd.DataFrame(res_test, index=X1.index, columns=cols)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        else:
                            continue
                    except:
                        continue
            elif el in ["OD", "BINOD", "OD-BINOD"]:
                level_cols = []
                for j, string_method in enumerate(self.list_methods_od):
                    print(string_method)
                    if "DBSCAN" in string_method:
                        cl, m = outlier_detection_class(
                            X1[self.dict_atts[i]],
                            string_method,
                            perc_train=self.perc_train,
                        )
                        try:
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in [
                        "PYOD_KNN_Largest",
                        "PYOD_KNN_Median",
                        "PYOD_KNN_Mean",
                        "SKL_LOF",
                    ]:
                        for k_num, k in enumerate(self.k_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["SKL_ISO_N"]:
                        for k_num, n in enumerate(self.n_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_N{}".format(n)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                    "bin_scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["PYOD_OCSVM_NU"]:
                        for k_num, nu in enumerate(self.nu_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_NU{}".format(nu)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                    "bin_scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method == "LOP_basic":
                        for k_num, k in enumerate(self.kl_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                scores_X = m.local_outlier_probabilities
                                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif "SKL_ELLENV" in string_method:
                        try:
                            m = self.dict_method[i][string_method]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X, maha_scores_X],
                                columns=cols,
                                index=X1.index,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    else:
                        try:
                            m = self.dict_method[i][string_method]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
            elif el in ["OD-RED", "BINOD-RED", "OD-BINOD-RED"]:
                level_cols = []
                for j, string_method in enumerate(self.list_methods_od):
                    # print(string_method)
                    if "DBSCAN" in string_method:
                        try:
                            cl, m = outlier_detection_class(
                                X1[self.dict_atts[i]],
                                string_method,
                                perc_train=self.perc_train,
                            )
                            cluster_labels_X = cl.fit_predict(X1[self.dict_atts[i]])
                            m.cluster_labels = cluster_labels_X
                        except:
                            continue
                        try:
                            m_X = m.fit()
                            scores_X = m_X.local_outlier_probabilities
                            bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    elif string_method in [
                        "PYOD_KNN_Largest",
                        "PYOD_KNN_Median",
                        "PYOD_KNN_Mean",
                        "SKL_LOF",
                    ]:
                        for k_num, k in enumerate(self.k_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_K{}".format(k)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["SKL_ISO_N"]:
                        for k_num, n in enumerate(self.n_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_N{}".format(n)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                    "bin_scores_{}_{}_N{}".format(
                                        colname, string_method, n
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method in ["PYOD_OCSVM_NU"]:
                        for k_num, nu in enumerate(self.nu_list):
                            try:
                                m = self.dict_method[i][
                                    string_method + "_NU{}".format(nu)
                                ]
                                bin_scores_X = m.predict(X1[self.dict_atts[i]])
                                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                                scores_X = m.decision_function(X1[self.dict_atts[i]])
                                cols = [
                                    "scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                    "bin_scores_{}_{}_NU{}".format(
                                        colname, string_method, nu
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif string_method == "LOP_basic":
                        for k_num, k in enumerate(self.kl_list):
                            try:
                                m = outlier_detection_class(
                                    X1[self.dict_atts[i]],
                                    string_method,
                                    k=k,
                                    perc_train=self.perc_train,
                                )
                                scores_X = m.local_outlier_probabilities
                                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                                cols = [
                                    "scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                    "bin_scores_{}_{}_K{}".format(
                                        colname, string_method, k
                                    ),
                                ]
                                tmp = pd.DataFrame(
                                    np.c_[scores_X, bin_scores_X],
                                    index=X1.index,
                                    columns=cols,
                                )
                                tmp = tmp.astype(float)
                                if X2 is None:
                                    X2 = pd.concat([X1, tmp], axis=1)
                                else:
                                    X2 = pd.concat([X2, tmp], axis=1)
                            except:
                                continue
                    elif "SKL_ELLENV" in string_method:
                        try:
                            m = self.dict_method[i][string_method]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            maha_scores_X = m.mahalanobis(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                                "mah_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X, maha_scores_X],
                                columns=cols,
                                index=X1.index,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                    else:
                        try:
                            m = self.dict_method[i][string_method]
                            bin_scores_X = m.predict(X1[self.dict_atts[i]])
                            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                            scores_X = m.decision_function(X1[self.dict_atts[i]])
                            cols = [
                                "scores_{}_{}".format(colname, string_method),
                                "bin_scores_{}_{}".format(colname, string_method),
                            ]
                            tmp = pd.DataFrame(
                                np.c_[scores_X, bin_scores_X],
                                index=X1.index,
                                columns=cols,
                            )
                            tmp = tmp.astype(float)
                            if X2 is None:
                                X2 = pd.concat([X1, tmp], axis=1)
                            else:
                                X2 = pd.concat([X2, tmp], axis=1)
                        except:
                            continue
                for j, string_method in enumerate(self.list_methods_red):
                    # print(string_method)
                    m = copy.deepcopy(self.dict_method[i][string_method])
                    # scores_train
                    if m is not None:
                        if string_method in ["TSNE", "MDS", "SE"]:
                            res_test = m.fit_transform(X1[self.dict_atts[i]])
                        else:
                            res_test = m.transform(X1[self.dict_atts[i]])
                        cols = [
                            "{}_comp_{}_{}".format(string_method, colname, comp)
                            for comp in range(m.n_components)
                        ]
                        level_cols += cols
                        tmp = pd.DataFrame(res_test, index=X1.index, columns=cols)
                        X2 = pd.concat([X2, tmp], axis=1)
                    else:
                        continue
            if el not in ["ORIG"]:
                X1 = X2.copy()
                X2 = None
        # X1[self.scaling_feats] = self.scaler.transform(X1[self.scaling_feats])
        X1 = X1.copy()
        # db_test = pd.DataFrame(res_test, index=X.index, columns=cols
        self.data_test = X1
        return X1

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.data_train


class CResampler(
    BaseEstimator,
    ClassifierMixin,
):
    def __init__(
        self,
        clf_base,
        sampler,
        sampling_strategy,
        binary=True,
        meta="SMOTE",
        filename="stars",
        write_train=False,
        write_test=False,
    ):
        self.clf_base = clf_base
        self.sampling_strategy = sampling_strategy
        self.sampler = sampler(sampling_strategy=sampling_strategy)
        self.binary = binary
        self.classes_ = [0, 1]
        self.time_to_expand = None
        self.filename = filename
        self.write_train = write_train
        self.write_test = write_test
        self.meta = "{}_{}".format(meta, sampling_strategy * 100)

    def fit(self, X, y, sample_weight=None):
        self.atts_ = X.columns.tolist()
        if self.binary:
            l, s = len(y), sum(y)
            minc = min(l - s, s)
            maxc = l - minc
            self.b = minc / maxc / self.sampling_strategy
        time_start = time.time()
        X_train, y_train = self.sampler.fit_resample(X, y)
        time_end = time.time()
        self.time_to_expand = time_end - time_start
        if self.write_train:
            if isinstance(X_train, pd.DataFrame):
                X_train.to_csv(
                    "data/{}/{}_{}_train.csv".format(
                        self.filename, self.filename, self.meta
                    ),
                    index=False,
                )
            else:
                df_train = pd.DataFrame(X_train, columns=self.atts_)
                df_train.to_csv(
                    "data/{}/{}_{}_train.csv".format(
                        self.filename, self.filename, self.meta
                    ),
                    index=False,
                )
        if sample_weight is None:
            self.clf_base.fit(X_train, y_train)
        else:
            self.clf_base.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if self.binary:
            proba1 = self.clf_base.predict_proba(X)[:, 1]
            scaled = self.b * proba1 / (1 + proba1 * (self.b - 1))
            return np.column_stack(((1 - scaled), scaled))
        else:
            return self.clf_base.predict_proba(X)


class CCure(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        clf_base,
        alpha=0.05,
        beta=0.5,
        std=1.15,
        binary=True,
        meta="CURE",
        filename="stars",
        write_train=False,
        write_test=False,
    ):
        self.clf_base = clf_base
        self.sampler = Cure()
        self.alpha = alpha
        self.beta = beta
        self.std = std
        self.binary = binary
        self.classes_ = [0, 1]
        self.time_to_expand = None
        self.filename = filename
        self.write_train = write_train
        self.write_test = write_test
        self.meta = "{}_{}".format(meta, beta * 100)

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            self.atts_ = X.columns.tolist()
            X = X.values.copy()
            y = y.values.copy()
        minCls = np.argmin(np.bincount(y.astype(int)))
        majCls = np.argmax(np.bincount(y.astype(int)))
        minSize = np.sum(y == minCls)
        majSize = np.sum(y == majCls)
        labels = np.unique(y)
        counts = [len(y[y == label]) for label in labels]
        minority_class = labels[np.argmin(counts)]
        time_start = time.time()
        self.sampler.fit(X, y, alpha=self.alpha, stds=self.std)
        newMajSize = np.round(self.beta * majSize)
        newMinSize = minSize + np.round((1 - self.beta) * majSize)
        newMinSize = np.min([newMinSize, newMajSize])
        X_train, y_train = self.sampler.resample(
            rsmpldSize=dict(
                {majCls: newMajSize.astype(int), minCls: newMinSize.astype(int)}
            )
        )
        if self.binary:
            l, s = len(y), sum(y)
            minc = min(l - s, s)
            maxc = l - minc
            self.b = minc / maxc / (newMinSize / newMajSize)
        time_end = time.time()
        self.time_to_expand = time_end - time_start
        if self.write_train:
            X_train = pd.DataFrame(X_train, columns=self.atts_)
            X_train.to_csv(
                "data/{}/{}_{}_train.csv".format(
                    self.filename, self.filename, self.meta
                ),
                index=False,
            )
        if sample_weight is None:
            self.clf_base.fit(X_train, y_train)
        else:
            self.clf_base.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if self.binary:
            proba1 = self.clf_base.predict_proba(X)[:, 1]
            scaled = self.b * proba1 / (1 + proba1 * (self.b - 1))
            return np.column_stack(((1 - scaled), scaled))
        else:
            return self.clf_base.predict_proba(X)
