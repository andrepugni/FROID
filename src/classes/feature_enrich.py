import numpy as np
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

# from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
import random


def dim_reduction_class(X, y, string_method):
    np.random.seed(42)
    random.seed(42)
    if string_method == "PCA":
        params_dict = {"n_components": 2}
        m = PCA(**params_dict, random_state=42)
    elif string_method == "RSP":
        params_dict = {"n_components": 2, "random_state": 42}
        m = random_projection.GaussianRandomProjection(**params_dict)
    elif string_method == "ISO":
        params_dict = {"n_components": 2}
        m = Isomap(**params_dict)
    elif string_method == "TSNE":
        params_dict = {"n_components": 2, "random_state": 42, "n_jobs": 2}
        m = TSNE(**params_dict)
    elif string_method == "LDA":
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        a_1 = min(n_features, n_classes - 1)
        a_2 = 10
        params_dict = {"n_components": min(a_1, a_2)}
        m = LinearDiscriminantAnalysis(**params_dict)
    elif string_method == "KPCA_RBF":
        params_dict = {
            "n_components": 2,
            "kernel": "rbf",
            "gamma": 15,
            "random_state": 42,
        }
        m = KernelPCA(**params_dict)
    elif string_method == "KPCA_COS":
        params_dict = {
            "n_components": 2,
            "kernel": "cosine",
            "gamma": 15,
            "random_state": 42,
        }
        m = KernelPCA(**params_dict)
    elif string_method == "KPCA_SIG":
        params_dict = {
            "n_components": 2,
            "kernel": "sigmoid",
            "gamma": 15,
            "random_state": 42,
            "n_jobs": 2,
        }
        m = KernelPCA(**params_dict)
    elif string_method == "KPCA_POLY":
        params_dict = {
            "n_components": 2,
            "kernel": "poly",
            "gamma": 15,
            "random_state": 42,
            "n_jobs": 2,
        }
        m = KernelPCA(**params_dict)
    elif string_method == "LLE":
        params_dict = {
            "n_neighbors": 50,
            "n_components": 2,
            "eigen_solver": "auto",
            "method": "standard",
            "random_state": 42,
            "n_jobs": 2,
        }
        m = manifold.LocallyLinearEmbedding(**params_dict)
    elif string_method == "LLE_HESS":
        params_dict = {
            "n_neighbors": 50,
            "n_components": 2,
            "eigen_solver": "dense",
            "method": "hessian",
            "random_state": 42,
            "n_jobs": 2,
        }
        m = manifold.LocallyLinearEmbedding(**params_dict)
    elif string_method == "LLE_MOD":
        params_dict = {
            "n_neighbors": 50,
            "n_components": 2,
            "eigen_solver": "dense",
            "method": "modified",
            "random_state": 42,
            "n_jobs": 2,
        }
        m = manifold.LocallyLinearEmbedding(**params_dict)
    elif string_method == "MDS":
        params_dict = {
            "max_iter": 100,
            "n_components": 2,
            "n_init": 1,
            "random_state": 42,
            "n_jobs": 2,
        }
        m = manifold.MDS(**params_dict)
    elif string_method == "SE":
        params_dict = {
            "n_neighbors": 50,
            "n_components": 2,
            "random_state": 42,
            "n_jobs": 2,
        }
        m = manifold.SpectralEmbedding(**params_dict)
    try:
        if string_method == "LDA":
            m.fit(X, y)
        else:
            m.fit(X)
        return m
    except:
        return None


k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]


def outlier_detection_class(
    X, string_method, prog_bar=False, perc_train=0.001, k=1, n=10, nu=0.1
):
    np.random.seed(42)
    random.seed(42)
    X = X.copy()
    if "SKL_ELLENV" in string_method:
        if string_method == "SKL_ELLENV_01":
            params_ellenv = {"contamination": 0.001}
        elif string_method == "SKL_ELLENV_02":
            params_ellenv = {"contamination": 0.01}
        elif string_method == "SKL_ELLENV_03":
            params_ellenv = {"contamination": 0.1}
        elif string_method == "SKL_ELLENV_04":
            params_ellenv = {"contamination": 0.2}
        elif string_method == "SKL_ELLENV_05":
            params_ellenv = {"contamination": 0.5}
        try:
            ee = EllipticEnvelope(random_state=42, **params_ellenv)
            ee.fit(X)
            return ee
        except:
            return None
    elif "DBSCAN" in string_method:
        if string_method == "DBSCAN_01":
            params_dbscan = {"eps": 0.1, "min_samples": 10, "n_jobs": 2}
            params_loop = {"extent": 1, "n_neighbors": 10}
        elif string_method == "DBSCAN_02":
            params_dbscan = {"eps": 0.1, "min_samples": 20, "n_jobs": 2}
            params_loop = {"extent": 2, "n_neighbors": 10}
        elif string_method == "DBSCAN_03":
            params_dbscan = {"eps": 0.1, "min_samples": 20, "n_jobs": 2}
            params_loop = {"extent": 2, "n_neighbors": 20}
        elif string_method == "DBSCAN_04":
            params_dbscan = {"eps": 0.1, "min_samples": 50, "n_jobs": 2}
            params_loop = {"extent": 3, "n_neighbors": 10}
        elif string_method == "DBSCAN_05":
            params_dbscan = {"eps": 0.1, "min_samples": 50, "n_jobs": 2}
            params_loop = {"extent": 3, "n_neighbors": 20}
        elif string_method == "DBSCAN_06":
            params_dbscan = {"eps": 0.1, "min_samples": 50, "n_jobs": 2}
            params_loop = {"extent": 3, "n_neighbors": 50}
        elif string_method == "DBSCAN_07":
            params_dbscan = {"eps": 0.3, "min_samples": 10, "n_jobs": 2}
            params_loop = {"extent": 1, "n_neighbors": 10}
        elif string_method == "DBSCAN_08":
            params_dbscan = {"eps": 0.3, "min_samples": 20, "n_jobs": 2}
            params_loop = {"extent": 2, "n_neighbors": 10}
        elif string_method == "DBSCAN_09":
            params_dbscan = {"eps": 0.3, "min_samples": 50, "n_jobs": 2}
            params_loop = {"extent": 3, "n_neighbors": 10}
        elif string_method == "DBSCAN_10":
            params_dbscan = {"eps": 0.3, "min_samples": 20, "n_jobs": 2}
            params_loop = {"extent": 2, "n_neighbors": 10}
        db = DBSCAN(**params_dbscan)
        m = loop.LocalOutlierProbability(X, progress_bar=False, **params_loop)
        return db, m

    else:
        if "LOP_basic" in string_method:
            clf = loop.LocalOutlierProbability(
                X, progress_bar=prog_bar, n_neighbors=k
            ).fit()
        elif "SKL_ISO" in string_method:
            if string_method == "SKL_ISO_01":
                params_ISO = {"contamination": 0.001}
            elif string_method == "SKL_ISO_02":
                params_ISO = {"contamination": 0.01}
            elif string_method == "SKL_ISO_03":
                params_ISO = {"contamination": 0.1}
            elif string_method == "SKL_ISO_04":
                params_ISO = {"contamination": 0.2}
            elif string_method == "SKL_ISO_05":
                params_ISO = {"contamination": 0.5}
            elif "SKL_ISO_N" in string_method:
                params_ISO = {"n_estimators": n}
            clf = IsolationForest(n_jobs=2, random_state=42, **params_ISO)
            clf.fit(X)
        elif ("SKL_LOF" in string_method) and (string_method != "SKL_LOF_def"):
            params_LOF = {"n_neighbors": k}
            clf = LocalOutlierFactor(
                n_jobs=2, novelty=True, metric="minkowski", **params_LOF
            )
            clf.fit(X)
        elif string_method == "SKL_LOF_def":
            params_LOF = {"n_neighbors": 20}
            clf = LocalOutlierFactor(
                n_jobs=2, novelty=True, metric="minkowski", **params_LOF
            )
            clf.fit(X)
        elif "PYOD_COPOD" in string_method:
            if string_method == "PYOD_COPOD_01":
                params_pyod = {"contamination": 0.001, "n_jobs": 2}
            elif string_method == "PYOD_COPOD_02":
                params_pyod = {"contamination": 0.01, "n_jobs": 2}
            elif string_method == "PYOD_COPOD_03":
                params_pyod = {"contamination": 0.1, "n_jobs": 2}
            elif string_method == "PYOD_COPOD_04":
                params_pyod = {"contamination": 0.2, "n_jobs": 2}
            elif string_method == "PYOD_COPOD_05":
                params_pyod = {"contamination": 0.5, "n_jobs": 2}
            clf = COPOD(**params_pyod)
            clf.fit(X)

        elif "PYOD_ECOD" in string_method:
            if string_method == "PYOD_ECOD_01":
                params_pyod = {"contamination": 0.001, "n_jobs": 2}
            elif string_method == "PYOD_ECOD_02":
                params_pyod = {"contamination": 0.01, "n_jobs": 2}
            elif string_method == "PYOD_ECOD_03":
                params_pyod = {"contamination": 0.1, "n_jobs": 2}
            elif string_method == "PYOD_ECOD_04":
                params_pyod = {"contamination": 0.2, "n_jobs": 2}
            elif string_method == "PYOD_ECOD_05":
                params_pyod = {"contamination": 0.5, "n_jobs": 2}
            clf = ECOD(**params_pyod)
            clf.fit(X)

        elif "PYOD_SOS" in string_method:
            if string_method == "PYOD_SOS_01":
                params_pyod = {"contamination": 0.001}
            elif string_method == "PYOD_SOS_02":
                params_pyod = {"contamination": 0.01}
            elif string_method == "PYOD_SOS_03":
                params_pyod = {"contamination": 0.1}
            elif string_method == "PYOD_SOS_04":
                params_pyod = {"contamination": 0.2}
            elif string_method == "PYOD_SOS_05":
                params_pyod = {"contamination": 0.5}
            clf = SOS(**params_pyod)
            clf.fit(X)

        elif "PYOD_OCSVM" in string_method:
            if string_method == "PYOD_OCSVM_01":
                params_pyod = {"contamination": 0.001, "nu": 0.001}
            elif string_method == "PYOD_OCSVM_02":
                params_pyod = {"contamination": 0.01, "nu": 0.001}
            elif string_method == "PYOD_OCSVM_03":
                params_pyod = {"contamination": 0.1, "nu": 0.001}
            elif string_method == "PYOD_OCSVM_04":
                params_pyod = {"contamination": 0.2, "nu": 0.001}
            elif string_method == "PYOD_OCSVM_05":
                params_pyod = {"contamination": 0.5, "nu": 0.001}
            elif string_method == "PYOD_OCSVM_06":
                params_pyod = {"contamination": 0.001, "nu": 0.1}
            elif string_method == "PYOD_OCSVM_07":
                params_pyod = {"contamination": 0.01, "nu": 0.1}
            elif string_method == "PYOD_OCSVM_08":
                params_pyod = {"contamination": 0.1, "nu": 0.1}
            elif string_method == "PYOD_OCSVM_09":
                params_pyod = {"contamination": 0.2, "nu": 0.1}
            elif string_method == "PYOD_OCSVM_10":
                params_pyod = {"contamination": 0.5, "nu": 0.1}
            elif string_method == "PYOD_OCSVM_11":
                params_pyod = {"contamination": 0.001, "nu": 0.3}
            elif string_method == "PYOD_OCSVM_12":
                params_pyod = {"contamination": 0.01, "nu": 0.3}
            elif string_method == "PYOD_OCSVM_13":
                params_pyod = {"contamination": 0.1, "nu": 0.3}
            elif string_method == "PYOD_OCSVM_14":
                params_pyod = {"contamination": 0.2, "nu": 0.3}
            elif string_method == "PYOD_OCSVM_15":
                params_pyod = {"contamination": 0.5, "nu": 0.3}
            elif string_method == "PYOD_OCSVM_16":
                params_pyod = {"contamination": 0.001, "nu": 0.5}
            elif string_method == "PYOD_OCSVM_17":
                params_pyod = {"contamination": 0.01, "nu": 0.5}
            elif string_method == "PYOD_OCSVM_18":
                params_pyod = {"contamination": 0.1, "nu": 0.5}
            elif string_method == "PYOD_OCSVM_19":
                params_pyod = {"contamination": 0.2, "nu": 0.5}
            elif string_method == "PYOD_OCSVM_20":
                params_pyod = {"contamination": 0.5, "nu": 0.5}
            elif string_method == "PYOD_OCSVM_NU":
                params_pyod = {"nu": nu}
            clf = OCSVM(**params_pyod)
            clf.fit(X)

        elif "PYOD_COF" in string_method:
            if string_method == "PYOD_COF_01":
                params_pyod = {"contamination": 0.001, "n_neighbors": 10}
            elif string_method == "PYOD_COF_02":
                params_pyod = {"contamination": 0.01, "n_neighbors": 10}
            elif string_method == "PYOD_COF_03":
                params_pyod = {"contamination": 0.1, "n_neighbors": 10}
            elif string_method == "PYOD_COF_04":
                params_pyod = {"contamination": 0.2, "n_neighbors": 10}
            elif string_method == "PYOD_COF_05":
                params_pyod = {"contamination": 0.5, "n_neighbors": 10}
            elif string_method == "PYOD_COF_06":
                params_pyod = {"contamination": 0.001, "n_neighbors": 20}
            elif string_method == "PYOD_COF_07":
                params_pyod = {"contamination": 0.01, "n_neighbors": 20}
            elif string_method == "PYOD_COF_08":
                params_pyod = {"contamination": 0.1, "n_neighbors": 20}
            elif string_method == "PYOD_COF_09":
                params_pyod = {"contamination": 0.2, "n_neighbors": 20}
            elif string_method == "PYOD_COF_10":
                params_pyod = {"contamination": 0.5, "n_neighbors": 20}
            elif string_method == "PYOD_COF_11":
                params_pyod = {"contamination": 0.001, "n_neighbors": 50}
            elif string_method == "PYOD_COF_12":
                params_pyod = {"contamination": 0.01, "n_neighbors": 50}
            elif string_method == "PYOD_COF_13":
                params_pyod = {"contamination": 0.1, "n_neighbors": 50}
            elif string_method == "PYOD_COF_14":
                params_pyod = {"contamination": 0.2, "n_neighbors": 50}
            elif string_method == "PYOD_COF_15":
                params_pyod = {"contamination": 0.5, "n_neighbors": 50}
            clf = COF(**params_pyod)
            clf.fit(X)

        elif "PYOD_CBLOF" in string_method:
            if string_method == "PYOD_CBLOF_01":
                params_pyod = {"contamination": 0.001, "n_jobs": 2}
            elif string_method == "PYOD_CBLOF_02":
                params_pyod = {"contamination": 0.01, "n_jobs": 2}
            elif string_method == "PYOD_CBLOF_03":
                params_pyod = {"contamination": 0.1, "n_jobs": 2}
            elif string_method == "PYOD_CBLOF_04":
                params_pyod = {"contamination": 0.2, "n_jobs": 2}
            elif string_method == "PYOD_CBLOF_05":
                params_pyod = {"contamination": 0.5, "n_jobs": 2}
            clf = CBLOF(random_state=42, **params_pyod)
            clf.fit(X)

        elif "PYOD_HBOS" in string_method:
            if string_method == "PYOD_HBOS_01":
                params_pyod = {"contamination": 0.001}
            elif string_method == "PYOD_HBOS_02":
                params_pyod = {"contamination": 0.01}
            elif string_method == "PYOD_HBOS_03":
                params_pyod = {"contamination": 0.1}
            elif string_method == "PYOD_HBOS_04":
                params_pyod = {"contamination": 0.2}
            elif string_method == "PYOD_HBOS_05":
                params_pyod = {"contamination": 0.5}
            clf = HBOS(**params_pyod)
            try:
                clf.fit(X)
            except AssertionError:
                clf = None

        elif "PYOD_KNN" in string_method:
            if string_method == "PYOD_KNN_01":
                params_pyod = {"contamination": 0.001, "n_jobs": 2}
            elif string_method == "PYOD_KNN_02":
                params_pyod = {"contamination": 0.01, "n_jobs": 2}
            elif string_method == "PYOD_KNN_03":
                params_pyod = {"contamination": 0.1, "n_jobs": 2}
            elif string_method == "PYOD_KNN_04":
                params_pyod = {"contamination": 0.2, "n_jobs": 2}
            elif string_method == "PYOD_KNN_05":
                params_pyod = {"contamination": 0.5, "n_jobs": 2}
            elif string_method == "PYOD_KNN_Median":
                params_pyod = {
                    "contamination": perc_train,
                    "n_jobs": 2,
                    "n_neighbors": k,
                    "method": "median",
                }
            elif string_method == "PYOD_KNN_Mean":
                params_pyod = {
                    "contamination": perc_train,
                    "n_jobs": 2,
                    "n_neighbors": k,
                    "method": "mean",
                }
            elif string_method == "PYOD_KNN_Largest":
                params_pyod = {
                    "contamination": perc_train,
                    "n_jobs": 2,
                    "n_neighbors": k,
                    "method": "largest",
                }

            clf = KNN(**params_pyod)
            clf.fit(X)

        elif "PYOD_FeatureBagging" in string_method:
            if string_method == "PYOD_FeatureBagging_01":
                params_pyod = {"contamination": 0.001, "n_jobs": 2}
            elif string_method == "PYOD_FeatureBagging_02":
                params_pyod = {"contamination": 0.01, "n_jobs": 2}
            elif string_method == "PYOD_FeatureBagging_03":
                params_pyod = {"contamination": 0.1, "n_jobs": 2}
            elif string_method == "PYOD_FeatureBagging_04":
                params_pyod = {"contamination": 0.2, "n_jobs": 2}
            elif string_method == "PYOD_FeatureBagging_05":
                params_pyod = {"contamination": 0.5, "n_jobs": 2}
            clf = FeatureBagging(random_state=42, **params_pyod)
            clf.fit(X)

        elif "PYOD_MCD" in string_method:
            if string_method == "PYOD_MCD_01":
                params_pyod = {"contamination": 0.001}
            elif string_method == "PYOD_MCD_02":
                params_pyod = {"contamination": 0.01}
            elif string_method == "PYOD_MCD_03":
                params_pyod = {"contamination": 0.1}
            elif string_method == "PYOD_MCD_04":
                params_pyod = {"contamination": 0.2}
            elif string_method == "PYOD_MCD_05":
                params_pyod = {"contamination": 0.5}
            clf = MCD(random_state=42, **params_pyod)
            clf.fit(X)

        elif "PYOD_LODA" in string_method:
            if string_method == "PYOD_LODA_01":
                params_pyod = {"contamination": 0.001}
            elif string_method == "PYOD_LODA_02":
                params_pyod = {"contamination": 0.01}
            elif string_method == "PYOD_LODA_03":
                params_pyod = {"contamination": 0.1}
            elif string_method == "PYOD_LODA_04":
                params_pyod = {"contamination": 0.2}
            elif string_method == "PYOD_LODA_05":
                params_pyod = {"contamination": 0.5}
            clf = LODA(**params_pyod)
            clf.fit(X)

        elif "PYOD_SUOD" in string_method:
            if string_method == "PYOD_SUOD_01":
                params_pyod = {"contamination": 0.001}
            elif string_method == "PYOD_SUOD_02":
                params_pyod = {"contamination": 0.01}
            elif string_method == "PYOD_SUOD_03":
                params_pyod = {"contamination": 0.1}
            elif string_method == "PYOD_SUOD_04":
                params_pyod = {"contamination": 0.2}
            elif string_method == "PYOD_SUOD_05":
                params_pyod = {"contamination": 0.5}
            clf = SUOD(**params_pyod)
            try:
                clf.fit(X)
            except:
                clf = None
        elif "PYOD_PCA" in string_method:
            if string_method == "PYOD_PCANW_01":
                params_pyod = {
                    "contamination": 0.001,
                    "n_components": 2,
                    "whiten": False,
                }
            elif string_method == "PYOD_PCANW_02":
                params_pyod = {
                    "contamination": 0.01,
                    "n_components": 2,
                    "whiten": False,
                }
            elif string_method == "PYOD_PCANW_03":
                params_pyod = {"contamination": 0.1, "n_components": 2, "whiten": False}
            elif string_method == "PYOD_PCANW_04":
                params_pyod = {"contamination": 0.2, "n_components": 2, "whiten": False}
            elif string_method == "PYOD_PCANW_05":
                params_pyod = {"contamination": 0.5, "n_components": 2, "whiten": False}
            elif string_method == "PYOD_PCAW_01":
                params_pyod = {
                    "contamination": 0.001,
                    "n_components": 2,
                    "whiten": True,
                }
            elif string_method == "PYOD_PCAW_02":
                params_pyod = {"contamination": 0.01, "n_components": 2, "whiten": True}
            elif string_method == "PYOD_PCAW_03":
                params_pyod = {"contamination": 0.1, "n_components": 2, "whiten": True}
            elif string_method == "PYOD_PCAW_04":
                params_pyod = {"contamination": 0.2, "n_components": 2, "whiten": True}
            elif string_method == "PYOD_PCAW_05":
                params_pyod = {"contamination": 0.5, "n_components": 2, "whiten": True}
            clf = PCAM(random_state=42, **params_pyod)
            clf.fit(X)
        """
        elif "PYOD_NET_AE" in string_method:
            if string_method == "PYOD_NET_AE_01":
                hidden_structure = [
                    int(X.shape[1]),
                    int(X.shape[1] / 2),
                    int(X.shape[1] / 2),
                    int(X.shape[1]),
                ]
                params_pyod = {
                    "contamination": 0.001,
                    "epochs": 1,
                    "hidden_neurons": hidden_structure,
                }
            if string_method == "PYOD_NET_AE_02":
                hidden_structure = [
                    int(X.shape[1]),
                    int(X.shape[1] / 2),
                    int(X.shape[1] / 2),
                    int(X.shape[1]),
                ]
                params_pyod = {
                    "contamination": 0.01,
                    "epochs": 1,
                    "hidden_neurons": hidden_structure,
                }
            if string_method == "PYOD_NET_AE_03":
                hidden_structure = [
                    int(X.shape[1]),
                    int(X.shape[1] / 2),
                    int(X.shape[1] / 2),
                    int(X.shape[1]),
                ]
                params_pyod = {
                    "contamination": 0.1,
                    "epochs": 1,
                    "hidden_neurons": hidden_structure,
                }
            if string_method == "PYOD_NET_AE_04":
                hidden_structure = [
                    int(X.shape[1]),
                    int(X.shape[1] / 2),
                    int(X.shape[1] / 2),
                    int(X.shape[1]),
                ]
                params_pyod = {
                    "contamination": 0.2,
                    "epochs": 1,
                    "hidden_neurons": hidden_structure,
                }
            if string_method == "PYOD_NET_AE_05":
                hidden_structure = [
                    int(X.shape[1]),
                    int(X.shape[1] / 2),
                    int(X.shape[1] / 2),
                    int(X.shape[1]),
                ]
                params_pyod = {
                    "contamination": 0.5,
                    "epochs": 1,
                    "hidden_neurons": hidden_structure,
                }
            clf = AutoEncoder(**params_pyod)
            clf.fit(X)
        """
        return clf
