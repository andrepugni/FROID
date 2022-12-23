import pandas as pd
import numpy as np
from src.classes.froid_old import FROID
from PyNomaly import loop
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.mcd import MCD
from pyod.models.suod import SUOD
from pyod.models.loda import LODA
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.sos import SOS
from pyod.models.pca import PCA as PCAM
from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
import lightgbm as lgb
from src.classes.utils import list_methods_od, list_methods_red
from src.classes.feature_enrich import dim_reduction_class, outlier_detection_class
import random


# def test_methods_reduction_fit(filename='bank'):
#     # df = pd.read_csv('data/stars/stars.csv')
#     np.random.seed(42)
#     random.seed(42)
#     df_train = pd.read_csv('data/{}/{}_train.csv'.format(filename, filename))
#     clf_base = lgb.LGBMClassifier(random_state=42)
#     features = [col for col in df_train.columns.tolist() if col not in ['TARGET', 'INDEX']]
#     lmod = []
#     lred = list_methods_red
#     clf = FROID(clf_base, 'RED', list_methods_od=lmod, list_methods_red=lred)
#     clf.fit(df_train[features], df_train['TARGET'])
#     X1 = pd.DataFrame()
#     for j, string_method in enumerate(lred):
#         m = clf.dict_method[0][string_method]
#         if m is not None:
#             # scores_train
#             if string_method in ['TSNE', 'MDS', 'SE']:
#                 res_train = m.fit_transform(df_train[clf.dict_atts[0]])
#             else:
#                 res_train = m.transform(df_train[clf.dict_atts[0]])
#             cols = ["{}_comp_{}".format(string_method, comp) for comp in
#                     range(m.n_components)]
#             tmp = pd.DataFrame(res_train, index=df_train.index, columns=cols)
#             if j == 0:
#                 print("there we go")
#                 X2 = pd.concat([X1, tmp], axis=1)
#             else:
#                 X2 = pd.concat([X2, tmp], axis=1)
#     X3 = pd.DataFrame()
#     for i, string_method in enumerate(lred):
#         m = dim_reduction_class(df_train[clf.dict_atts[0]], df_train['TARGET'], string_method)
#         if m is not None:
#             # scores_train
#             if string_method in ['TSNE', 'MDS', 'SE']:
#                 res_train = m.fit_transform(df_train[clf.dict_atts[0]])
#             else:
#                 res_train = m.transform(df_train[clf.dict_atts[0]])
#             cols = ["{}_comp_{}".format(string_method, comp) for comp in
#                     range(m.n_components)]
#             tmp = pd.DataFrame(res_train, index=df_train.index, columns=cols)
#             if i == 0:
#                 print("there we go")
#                 X4 = pd.concat([X3, tmp], axis=1)
#             else:
#                 X4 = pd.concat([X4, tmp], axis=1)
#     assert np.testing.assert_allclose(X4.values, X2.values) is None
#
# def test_methods_od_fit(filename='bank'):
#     np.random.seed(42)
#     random.seed(42)
#     # df = pd.read_csv('data/stars/stars.csv')
#     df_train = pd.read_csv('data/{}/{}_train.csv'.format(filename, filename))
#     clf_base = lgb.LGBMClassifier(random_state=42)
#     features = [col for col in df_train.columns.tolist() if col not in ['TARGET', 'INDEX']]
#     lmod = list_methods_od
#     lred = []
#     clf = FROID(clf_base, 'OD-BINOD', list_methods_od=lmod, list_methods_red=lred)
#     clf.fit(df_train[features], df_train['TARGET'])
#     X1 = pd.DataFrame()
#     k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
#                   100, 150, 200, 250]
#     k_list = [k for k in k_list_pre if k < df_train.shape[0]]
#     kl_list_pre = [1, 5, 10, 20]
#     kl_list = [k for k in kl_list_pre if k < df_train.shape[0]]
#     n_list = [10, 20, 50, 70, 100, 150, 200, 250]
#     X1 = pd.DataFrame()
#     for j, string_method in enumerate(lmod):
#         print(string_method)
#         if 'DBSCAN' in string_method:
#             cl, m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method,
#                                             perc_train=clf.perc_train)
#             try:
#                 cluster_labels_X = cl.fit_predict(df_train[clf.dict_atts[0]])
#                 m.cluster_labels = cluster_labels_X
#             except:
#                 continue
#             try:
#                 m_X = m.fit()
#                 scores_X = m_X.local_outlier_probabilities
#                 bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
#                 cols = ["scores_{}".format(string_method),
#                         "bin_scores_{}".format(string_method)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if j == 0:
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#             except:
#                 continue
#         elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
#             for k_num, k in enumerate(k_list):
#                 m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, k=k,
#                                             perc_train=clf.perc_train)
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_K{}".format(string_method, k),
#                         "bin_scores_{}_K{}".format(string_method, k)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#         elif string_method in ['SKL_ISO_N']:
#             for k_num, n in enumerate(n_list):
#                 m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, n=n,
#                                             perc_train=clf.perc_train)
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_N{}".format(string_method, n),
#                         "bin_scores_{}_N{}".format(string_method, n)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#         elif string_method in ['PYOD_OCSVM_NU']:
#             for k_num, nu in enumerate(clf.nu_list):
#                 m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, nu=nu,
#                                             perc_train=clf.perc_train)
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_NU{}".format(string_method, nu),
#                         "bin_scores_{}_NU{}".format(string_method, nu)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#         elif string_method == 'LOP_basic':
#             for k_num, k in enumerate(clf.kl_list):
#                 m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, k=k,
#                                             perc_train=clf.perc_train)
#                 scores_X = m.local_outlier_probabilities
#                 bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
#                 cols = ["scores_{}_K{}".format(string_method, k),
#                         "bin_scores_{}_K{}".format(string_method, k)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#         elif 'SKL_ELLENV' in string_method:
#             m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, perc_train=clf.perc_train)
#             bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#             bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#             scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#             maha_scores_X = m.mahalanobis(df_train[clf.dict_atts[0]])
#             cols = ["scores_{}".format(string_method),
#                     "bin_scores_{}".format(string_method),
#                     "mah_scores_{}".format(string_method)]
#             tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
#                                index=df_train.index)
#             tmp = tmp.astype(float)
#             if j == 0:
#                 X2 = pd.concat([X1, tmp], axis=1)
#             else:
#                 X2 = pd.concat([X2, tmp], axis=1)
#         else:
#
#             m = outlier_detection_class(df_train[clf.dict_atts[0]], string_method, perc_train=clf.perc_train)
#             if m is not None:
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}".format(string_method),
#                         "bin_scores_{}".format(string_method)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if j == 0:
#                     X2 = pd.concat([X1, tmp], axis=1)
#                 else:
#                     X2 = pd.concat([X2, tmp], axis=1)
#     print("up to here all right")
#     X3 = pd.DataFrame()
#     for j, string_method in enumerate(lmod):
#         print(string_method)
#         if 'DBSCAN' in string_method:
#             try:
#                 cl, m = clf.dict_method[0][string_method]
#             except:
#                 continue
#             try:
#                 cluster_labels_X = cl.fit_predict(df_train[clf.dict_atts[0]])
#                 m.cluster_labels = cluster_labels_X
#             except:
#                 continue
#             try:
#                 m_X = m.fit()
#                 scores_X = m_X.local_outlier_probabilities
#                 bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
#                 cols = ["scores_{}".format(string_method),
#                         "bin_scores_{}".format(string_method)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if j == 0:
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#             except:
#                 continue
#         elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
#             for k_num, k in enumerate(k_list):
#                 m = clf.dict_method[0][string_method + "_K{}".format(k)]
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_K{}".format(string_method, k),
#                         "bin_scores_{}_K{}".format(string_method, k)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#         elif string_method in ['SKL_ISO_N']:
#             for k_num, n in enumerate(n_list):
#                 m = clf.dict_method[0][string_method + "_N{}".format(n)]
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_N{}".format(string_method, n),
#                         "bin_scores_{}_N{}".format(string_method, n)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#         elif string_method in ['PYOD_OCSVM_NU']:
#             for k_num, nu in enumerate(clf.nu_list):
#                 m = clf.dict_method[0][string_method + "_NU{}".format(nu)]
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}_NU{}".format(string_method, nu),
#                         "bin_scores_{}_NU{}".format(string_method, nu)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#         elif string_method == 'LOP_basic':
#             for k_num, k in enumerate(clf.kl_list):
#                 m = clf.dict_method[0][string_method + "_K{}".format(k)]
#                 scores_X = m.local_outlier_probabilities
#                 bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
#                 cols = ["scores_{}_K{}".format(string_method, k),
#                         "bin_scores_{}_K{}".format(string_method, k)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if (j == 0) & (k_num == 0):
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#         elif 'SKL_ELLENV' in string_method:
#             m = clf.dict_method[0][string_method]
#             bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#             bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#             scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#             maha_scores_X = m.mahalanobis(df_train[clf.dict_atts[0]])
#             cols = ["scores_{}".format(string_method),
#                     "bin_scores_{}".format(string_method),
#                     "mah_scores_{}".format(string_method)]
#             tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
#                                index=df_train.index)
#             tmp = tmp.astype(float)
#             if j == 0:
#                 X4 = pd.concat([X3, tmp], axis=1)
#             else:
#                 X4 = pd.concat([X4, tmp], axis=1)
#         else:
#             m = clf.dict_method[0][string_method]
#             if m is not None:
#                 bin_scores_X = m.predict(df_train[clf.dict_atts[0]])
#                 bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
#                 scores_X = m.decision_function(df_train[clf.dict_atts[0]])
#                 cols = ["scores_{}".format(string_method),
#                         "bin_scores_{}".format(string_method)]
#                 tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
#                 tmp = tmp.astype(float)
#                 if j == 0:
#                     X4 = pd.concat([X3, tmp], axis=1)
#                 else:
#                     X4 = pd.concat([X4, tmp], axis=1)
#     assert np.testing.assert_allclose(X4.values, X2.values) is None

def test_method_alldataset_vstrain(filename='bank'):
    np.random.seed(42)
    random.seed(42)
    df = pd.read_csv('data/{}/{}_all.csv'.format(filename, filename))
    df_train = pd.read_csv('data/{}/{}_train.csv'.format(filename, filename))
    # df_test = pd.read_csv('data/stars/stars_test.csv')
    clf_base = lgb.LGBMClassifier(random_state=42)
    features = [col for col in df_train.columns.tolist() if col not in ['TARGET', 'INDEX']]
    index_list = df_train.INDEX.values.tolist()
    lmod = list_methods_od[:4]
    lred = []
    clf1 = FROID(clf_base, 'OD-BINOD', list_methods_od=lmod, list_methods_red=lred)
    clf2 = FROID(clf_base, 'OD-BINOD', list_methods_od=lmod, list_methods_red=lred)
    clf1.fit(df_train[features], df_train['TARGET'])
    clf2.fit(df[features], df['TARGET'])
    k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 150, 200, 250]
    k_list = [k for k in k_list_pre if k < df_train.shape[0]]
    kl_list_pre = [1, 5, 10, 20]
    kl_list = [k for k in kl_list_pre if k < df_train.shape[0]]
    n_list = [10, 20, 50, 70, 100, 150, 200, 250]
    X1 = pd.DataFrame()
    for j, string_method in enumerate(lmod):
        print(string_method)
        if 'DBSCAN' in string_method:
            try:
                cl, m = clf1.dict_method[0][string_method]
            except:
                continue
            try:
                cluster_labels_X = cl.fit_predict(df_train[clf1.dict_atts[0]])
                m.cluster_labels = cluster_labels_X
            except:
                continue
            try:
                m_X = m.fit()
                scores_X = m_X.local_outlier_probabilities
                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                cols = ["scores_{}".format(string_method),
                        "bin_scores_{}".format(string_method)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if j == 0:
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
            except:
                continue
        elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
            for k_num, k in enumerate(k_list):
                m = clf1.dict_method[0][string_method + "_K{}".format(k)]
                bin_scores_X = m.predict(df_train[clf1.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf1.dict_atts[0]])
                cols = ["scores_{}_K{}".format(string_method, k),
                        "bin_scores_{}_K{}".format(string_method, k)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
        elif string_method in ['SKL_ISO_N']:
            for k_num, n in enumerate(n_list):
                m = clf1.dict_method[0][string_method + "_N{}".format(n)]
                bin_scores_X = m.predict(df_train[clf1.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf1.dict_atts[0]])
                cols = ["scores_{}_N{}".format(string_method, n),
                        "bin_scores_{}_N{}".format(string_method, n)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
        elif string_method in ['PYOD_OCSVM_NU']:
            for k_num, nu in enumerate(clf1.nu_list):
                m = clf1.dict_method[0][string_method + "_NU{}".format(nu)]
                bin_scores_X = m.predict(df_train[clf1.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf1.dict_atts[0]])
                cols = ["scores_{}_NU{}".format(string_method, nu),
                        "bin_scores_{}_NU{}".format(string_method, nu)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
        elif string_method == 'LOP_basic':
            for k_num, k in enumerate(clf1.kl_list):
                m = clf1.dict_method[0][string_method + "_K{}".format(k)]
                scores_X = m.local_outlier_probabilities
                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                cols = ["scores_{}_K{}".format(string_method, k),
                        "bin_scores_{}_K{}".format(string_method, k)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
        elif 'SKL_ELLENV' in string_method:
            m = clf1.dict_method[0][string_method]
            bin_scores_X = m.predict(df_train[clf1.dict_atts[0]])
            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
            scores_X = m.decision_function(df_train[clf1.dict_atts[0]])
            maha_scores_X = m.mahalanobis(df_train[clf1.dict_atts[0]])
            cols = ["scores_{}".format(string_method),
                    "bin_scores_{}".format(string_method),
                    "mah_scores_{}".format(string_method)]
            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                               index=df_train.index)
            tmp = tmp.astype(float)
            if j == 0:
                X2 = pd.concat([X1, tmp], axis=1)
            else:
                X2 = pd.concat([X2, tmp], axis=1)
        else:
            m = clf1.dict_method[0][string_method]
            if m is not None:
                bin_scores_X = m.predict(df_train[clf1.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf1.dict_atts[0]])
                cols = ["scores_{}".format(string_method),
                        "bin_scores_{}".format(string_method)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if j == 0:
                    X2 = pd.concat([X1, tmp], axis=1)
                else:
                    X2 = pd.concat([X2, tmp], axis=1)
    print("up to here all right")
    X3 = pd.DataFrame()
    for j, string_method in enumerate(lmod):
        print(string_method)
        if 'DBSCAN' in string_method:
            try:
                cl, m = clf2.dict_method[0][string_method]
            except:
                continue
            try:
                cluster_labels_X = cl.fit_predict(df_train[clf2.dict_atts[0]])
                m.cluster_labels = cluster_labels_X
            except:
                continue
            try:
                m_X = m.fit()
                scores_X = m_X.local_outlier_probabilities
                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                cols = ["scores_{}".format(string_method),
                        "bin_scores_{}".format(string_method)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if j == 0:
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
            except:
                continue
        elif string_method in ['PYOD_KNN_Largest', 'PYOD_KNN_Median', 'PYOD_KNN_Mean', 'SKL_LOF']:
            for k_num, k in enumerate(k_list):
                m = clf2.dict_method[0][string_method + "_K{}".format(k)]
                bin_scores_X = m.predict(df_train[clf2.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf2.dict_atts[0]])
                cols = ["scores_{}_K{}".format(string_method, k),
                        "bin_scores_{}_K{}".format(string_method, k)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
        elif string_method in ['SKL_ISO_N']:
            for k_num, n in enumerate(n_list):
                m = clf2.dict_method[0][string_method + "_N{}".format(n)]
                bin_scores_X = m.predict(df_train[clf2.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf2.dict_atts[0]])
                cols = ["scores_{}_N{}".format(string_method, n),
                        "bin_scores_{}_N{}".format(string_method, n)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
        elif string_method in ['PYOD_OCSVM_NU']:
            for k_num, nu in enumerate(clf2.nu_list):
                m = clf2.dict_method[0][string_method + "_NU{}".format(nu)]
                bin_scores_X = m.predict(df_train[clf2.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf2.dict_atts[0]])
                cols = ["scores_{}_NU{}".format(string_method, nu),
                        "bin_scores_{}_NU{}".format(string_method, nu)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
        elif string_method == 'LOP_basic':
            for k_num, k in enumerate(clf2.kl_list):
                m = clf2.dict_method[0][string_method + "_K{}".format(k)]
                scores_X = m.local_outlier_probabilities
                bin_scores_X = np.where(scores_X > 1 / 2, 1, 0)
                cols = ["scores_{}_K{}".format(string_method, k),
                        "bin_scores_{}_K{}".format(string_method, k)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if (j == 0) & (k_num == 0):
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
        elif 'SKL_ELLENV' in string_method:
            m = clf2.dict_method[0][string_method]
            bin_scores_X = m.predict(df_train[clf2.dict_atts[0]])
            bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
            scores_X = m.decision_function(df_train[clf2.dict_atts[0]])
            maha_scores_X = m.mahalanobis(df_train[clf2.dict_atts[0]])
            cols = ["scores_{}".format(string_method),
                    "bin_scores_{}".format(string_method),
                    "mah_scores_{}".format(string_method)]
            tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X, maha_scores_X], columns=cols,
                               index=df_train.index)
            tmp = tmp.astype(float)
            if j == 0:
                X4 = pd.concat([X3, tmp], axis=1)
            else:
                X4 = pd.concat([X4, tmp], axis=1)
        else:
            m = clf2.dict_method[0][string_method]
            if m is not None:
                bin_scores_X = m.predict(df_train[clf2.dict_atts[0]])
                bin_scores_X = np.where(bin_scores_X == 1, 1, 0)
                scores_X = m.decision_function(df_train[clf2.dict_atts[0]])
                cols = ["scores_{}".format(string_method),
                        "bin_scores_{}".format(string_method)]
                tmp = pd.DataFrame(np.c_[scores_X, bin_scores_X], index=df_train.index, columns=cols)
                tmp = tmp.astype(float)
                if j == 0:
                    X4 = pd.concat([X3, tmp], axis=1)
                else:
                    X4 = pd.concat([X4, tmp], axis=1)
    print(X4.shape)
    X4 = X4[X4.index.isin(index_list)].copy()
    X4 = X4.loc[index_list,:].copy()
    print(X4.shape)
    different = False
    try:
        np.testing.assert_allclose(X4.values, X2.values)
    except AssertionError:
        different = True
    assert different is False

