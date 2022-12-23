import pandas as pd
import numpy as np
import os

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from imblearn import metrics as imm
from sklearn import metrics as skm
import time
from sklearn.preprocessing import RobustScaler
from src.classes.froid import FROID, CResampler, CCure
from src.classes.resampling_schemes import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    SVMSMOTE,
    KMeansSMOTE,
)
import random
from xgboost import XGBClassifier
# from resampling_schemes import *
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import joblib


import pickle

list_methods_od = [
    "SKL_ISO_01",
    "SKL_ISO_02",
    "SKL_ISO_03",
    "SKL_ISO_04",
    "SKL_ISO_05",
    "SKL_ELLENV_01",
    "SKL_ELLENV_02",
    "SKL_ELLENV_03",
    "SKL_ELLENV_04",
    "SKL_ELLENV_05",
    "LOP_basic",
    "DBSCAN_01",
    "DBSCAN_02",
    "DBSCAN_03",
    "DBSCAN_04",
    "DBSCAN_05",
    "DBSCAN_06",
    "DBSCAN_07",
    "DBSCAN_08",
    "DBSCAN_09",
    "DBSCAN_10",
    "SKL_LOF",
    "PYOD_COPOD_01",
    "PYOD_COPOD_02",
    "PYOD_COPOD_03",
    "PYOD_COPOD_04",
    "PYOD_COPOD_05",
    "PYOD_ECOD_01",
    "PYOD_ECOD_02",
    "PYOD_ECOD_03",
    "PYOD_ECOD_04",
    "PYOD_ECOD_05",
    "PYOD_PCAW_01",
    "PYOD_PCAW_02",
    "PYOD_PCAW_03",
    "PYOD_PCAW_04",
    "PYOD_PCAW_05",
    "PYOD_OCSVM_01",
    "PYOD_OCSVM_02",
    "PYOD_OCSVM_03",
    "PYOD_OCSVM_04",
    "PYOD_OCSVM_05",
    "PYOD_OCSVM_06",
    "PYOD_OCSVM_07",
    "PYOD_OCSVM_08",
    "PYOD_OCSVM_09",
    "PYOD_OCSVM_10",
    "PYOD_OCSVM_11",
    "PYOD_OCSVM_12",
    "PYOD_OCSVM_13",
    "PYOD_OCSVM_14",
    "PYOD_OCSVM_15",
    "PYOD_OCSVM_16",
    "PYOD_OCSVM_17",
    "PYOD_OCSVM_18",
    "PYOD_OCSVM_19",
    "PYOD_OCSVM_20",
    "PYOD_COF_01",
    "PYOD_COF_02",
    "PYOD_COF_03",
    "PYOD_COF_04",
    "PYOD_COF_05",
    "PYOD_COF_06",
    "PYOD_COF_07",
    "PYOD_COF_08",
    "PYOD_COF_09",
    "PYOD_COF_10",
    "PYOD_COF_11",
    "PYOD_COF_12",
    "PYOD_COF_13",
    "PYOD_COF_14",
    "PYOD_COF_15",
    "PYOD_PCANW_01",
    "PYOD_PCANW_02",
    "PYOD_PCANW_03",
    "PYOD_PCANW_04",
    "PYOD_PCANW_05",
    "PYOD_CBLOF_01",
    "PYOD_CBLOF_02",
    "PYOD_CBLOF_03",
    "PYOD_CBLOF_04",
    "PYOD_CBLOF_05",
    "PYOD_HBOS_01",
    "PYOD_HBOS_02",
    "PYOD_HBOS_03",
    "PYOD_HBOS_04",
    "PYOD_HBOS_05",
    "PYOD_KNN_01",
    "PYOD_KNN_02",
    "PYOD_KNN_03",
    "PYOD_KNN_04",
    "PYOD_KNN_05",
    "PYOD_FeatureBagging_01",
    "PYOD_FeatureBagging_02",
    "PYOD_FeatureBagging_03",
    "PYOD_FeatureBagging_04",
    "PYOD_FeatureBagging_05",
    "PYOD_MCD_01",
    "PYOD_MCD_02",
    "PYOD_MCD_03",
    "PYOD_MCD_04",
    "PYOD_MCD_05",
    "PYOD_LODA_01",
    "PYOD_LODA_02",
    "PYOD_LODA_03",
    "PYOD_LODA_04",
    "PYOD_LODA_05",
    "PYOD_SUOD_01",
    "PYOD_SUOD_02",
    "PYOD_SUOD_03",
    "PYOD_SUOD_04",
    "PYOD_SUOD_05",
]
# kNN, LOF, COF, CBLOF,
#EllEnv, MCD, HBOS, COPOD, ECOD, FeaBag, IsoFor, LODA, SUOD, and
#PCA as OD functions Θ, and on GRP, PCA, t-SNE, LDA as FR functions P
list_methods_od_light = [
    "PYOD_KNN_01",
    "SKL_LOF_def",
    "PYOD_COF_05",
    "PYOD_CBLOF_03",
    "SKL_ELLENV_03",
    "PYOD_MCD_01",
    "PYOD_HBOS_01",
    "PYOD_SUOD_01",
    "PYOD_LODA_05",
    "PYOD_FeatureBagging_02",
    "PYOD_ISO_01",
    "PYOD_LODA_03",
    "PYOD_COPOD_01",

]
list_methods_xgbod = [
    "PYOD_KNN_Largest",
    "PYOD_KNN_Median",
    "PYOD_KNN_Mean",
    "SKL_LOF",
    "SKL_ISO_N",
    "PYOD_OCSVM_NU",
    "LOP_basic",
]
list_methods_red = [
    "ISO",
    "PCA",
    "KPCA_SIG",
    "KPCA_POLY",
    "KPCA_COS",
    "KPCA_RBF",
    "TSNE",
    "RSP",
    "LLE",
    "LLE_MOD",
    "LLE_HESS",
    "SE",
    "MDS",
]

list_methods_red_light = [
"ISO",
"PCA",
"TSNE"
"RSP"

]

def prepare_dataset_train_test(filename, atts_dict, target_dict, tounb_dict, scaling_dict, data_fold='data'):
    # here we create the path for the dataset
    data_path = "{}/{}/{}.csv".format(data_fold, filename, filename)
    # here we read the dataset
    if (os.path.exists("{}/{}/{}_train.csv".format(data_fold, filename, filename)))& \
        (os.path.exists("{}/{}/{}_test.csv".format(data_fold, filename, filename))):
        df_train = pd.read_csv("{}/{}/{}_train.csv".format(data_fold, filename, filename),)
        df_test = pd.read_csv("{}/{}/{}_test.csv".format(data_fold, filename, filename))
        print(df_train['TARGET'].value_counts())
        print(df_test['TARGET'].value_counts())
    else:
        df = pd.read_csv(data_path)
        if filename == 'stars':
            df['TARGET'] = df[target_dict[filename]].copy()
        elif filename == 'mammography':
            df['TARGET'] = np.where(df['class'] == "'-1'", 0, 1)
        # here we select the features
        atts = atts_dict[filename]
        all_cols = atts_dict[filename]
        all_cols.append('TARGET')
        df = df[all_cols]
        print('Null Values in DataSet:')
        print(df.isnull().sum())
        print('\nRows: ', df.shape[0])
        print('Columns: ', df.shape[1])
        print(df.dtypes)
        # df_1.select_dtypes(exclude=['int','float'])
        print('--- --- ---')
        print(df['TARGET'].value_counts())
        print(df.info())
        if tounb_dict[filename]:
            if filename == 'stars':
                df_pos = df[df['TARGET'] == 1].copy()
                df_neg = df[df['TARGET'] == 0].copy()
                df_neg = df_neg.sample(frac=.4, replace=False, random_state=42)
                df = pd.concat([df_neg,df_pos], axis=0)
                print(df['TARGET'].value_counts())
        # here we divide dataset in training and test set
        df_train, df_test = train_test_split(df,test_size=.3,stratify=df['TARGET'], random_state=42)
        # scale data
        if scaling_dict[filename]:
            print("scaling data")
            scaler = RobustScaler()
            scaler.fit(df_train[atts])
            df_train = df_train.reset_index(drop=False)
            df_test = df_test.reset_index(drop=False)
            df_train_sc = pd.DataFrame(scaler.transform(df_train[atts]), columns=atts)
            df_train_sc['TARGET'] = df_train['TARGET'].copy()
            df_train_sc['INDEX'] = df_train['index'].copy()
            print(df_train_sc.columns)
            df_test_sc = pd.DataFrame(scaler.transform(df_test[atts]), columns=atts)
            df_test_sc['TARGET'] = df_test['TARGET'].copy()
            df_test_sc['INDEX'] = df_test['index'].copy()
            print(df_test_sc.columns)
            df_train_sc.to_csv('{}/{}/{}_train.csv'.format(data_fold, filename,filename), index=False)
            df_test_sc.to_csv('{}/{}/{}_test.csv'.format(data_fold, filename,filename), index=False)
            # print(df_train_sc['TARGET'].value_counts())
            # print(df_test_sc['TARGET'].value_counts())
        else:
            df_train = df_train.reset_index(drop=False)
            df_test = df_test.reset_index(drop=False)
            df_train['INDEX'] = df_train['index'].copy()
            df_train.drop('index', axis=1,  inplace=True)
            df_test['INDEX'] = df_test['index'].copy()
            df_test.drop('index', axis=1,  inplace=True)
            df_train.to_csv('{}/{}/{}_train.csv'.format(data_fold, filename,filename), index=False)
            df_test.to_csv('{}/{}/{}_test.csv'.format(data_fold, filename,filename), index=False)
        print(df_train['TARGET'].value_counts())
        print(df_test['TARGET'].value_counts())

def get_dict_methods_od(data_fold="data", method='OD-BINOD-RED', method_list='default'):
    filenames = os.listdir(data_fold)
    print(method_list)
    print(data_fold)
    if method_list == 'default':
        dict_methods_od_files = {filename: list_methods_od.copy() for filename in filenames}
        dict_methods_od_files["yeast-0-2-5-7-9_vs_3-6-8"] = [
            el
            for el in dict_methods_od_files["yeast-0-2-5-7-9_vs_3-6-8"]
            if el not in ["DBSCAN_07", "DBSCAN_08", "DBSCAN_10"]
        ]
        dict_methods_od_files["solar_flare_m0"] = [
            el for el in dict_methods_od_files["solar_flare_m0"] if el not in ["LOP_basic"]
        ]
        dict_methods_od_files["yeast-0-3-5-9_vs_7-8"] = [
            el
            for el in dict_methods_od_files["yeast-0-3-5-9_vs_7-8"]
            if el not in ["DBSCAN_07", "DBSCAN_08", "DBSCAN_10"]
        ]
        dict_methods_od_files["yeast6"] = [
            el
            for el in dict_methods_od_files["yeast6"]
            if el not in ["DBSCAN_07", "DBSCAN_03", "DBSCAN_08", "DBSCAN_10"]
        ]
        dict_methods_od_files["letter_img"] = [
            el for el in dict_methods_od_files["letter_img"] if el not in ["DBSCAN_01", "DBSCAN_07"]
        ]
        dict_methods_od_files["led7digit-0-2-4-5-6-7-8-9_vs_1"] = [
            el for el in dict_methods_od_files["led7digit-0-2-4-5-6-7-8-9_vs_1"]if el not in ["DBSCAN_01", "DBSCAN_07"]
        ]
        dict_methods_od_files["protein_homo"] = [
            el
            for el in list_methods_od
            if el
            not in [
                "LOP_basic",
                "DBSCAN_01",
                "DBSCAN_02",
                "DBSCAN_03",
                "DBSCAN_04",
                "DBSCAN_05",
                "DBSCAN_06",
                "DBSCAN_07",
                "DBSCAN_08",
                "DBSCAN_09",
                "DBSCAN_10",
            ]
        ]
        dict_methods_od_files["ecoli4"] = [
            el for el in dict_methods_od_files["ecoli4"] if el not in ["DBSCAN_07"]
        ]
        dict_methods_od_files["yeast-2_vs_8"] = [
            el
            for el in dict_methods_od_files["yeast-2_vs_8"]
            if el not in ["DBSCAN_01", "DBSCAN_07", "DBSCAN_08", "DBSCAN_09", "DBSCAN_10"]
        ]

        dict_methods_od_files["kddcup-guess_passwd_vs_satan"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
        dict_methods_od_files["yeast-0-2-5-7-9_vs_3-6-8"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
        dict_methods_od_files["yeast-0-2-5-6_vs_3-7-8-9"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
        dict_methods_od_files["yeast-0-2-5-6_vs_3-7-8-9"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
        dict_methods_od_files["kddcup-land_vs_satan"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
        dict_methods_od_files["yeast-1-2-8-9_vs_7"] = [
            el for el in list_methods_od if el not in ["DBSCAN_07"]
        ]
        dict_methods_od_files["car_eval_4"] = [
            el
            for el in list_methods_od
            if el
            not in [
                "PYOD_CBLOF_01",
                "PYOD_CBLOF_02",
                "PYOD_CBLOF_03",
                "PYOD_CBLOF_04",
                "PYOD_CBLOF_05",
            ]
        ]
        dict_methods_od_files["poker-8-9_vs_5"] = [
            el
            for el in list_methods_od
            if el
            not in [
                "PYOD_CBLOF_01",
                "PYOD_CBLOF_02",
                "PYOD_CBLOF_03",
                "PYOD_CBLOF_04",
                "PYOD_CBLOF_05",
            ]
        ]
        dict_methods_od_files["abalone9-18"] = [
            el for el in list_methods_od if el not in ["DBSCAN_03"]
        ]
    elif method_list == 'xgbod':
        print("IDLES are the best")
        dict_methods_od_files = {filename: list_methods_xgbod.copy() for filename in filenames}
    elif method_list == 'light':
        print("Bon Iver are the best")
        dict_methods_od_files = {filename: list_methods_od_light.copy() for filename in filenames}
    if (method == "OD-BINOD^OD-BINOD") & (data_fold == 'data'):
        dict_methods_od_files["solar_flare_m0"] = [
            el for el in dict_methods_od_files['solar_flare_m0'] if el not in ["LOP_basic"]
        ]
        dict_methods_od_files["wine_quality"] = [
            el
            for el in dict_methods_od_files["wine_quality"]
            if el not in ["LOP_basic", "DBSCAN_03", "DBSCAN_05", "DBSCAN_06"]
        ]
        dict_methods_od_files["kddcup-land_vs_satan"] = [
            el
            for el in dict_methods_od_files["kddcup-land_vs_satan"]
            if el
            not in [
                "LOP_basic",
                "DBSCAN_02",
                "DBSCAN_04",
                "DBSCAN_05",
                "DBSCAN_06",
                "DBSCAN_08",
                "DBSCAN_09",
                "DBSCAN_10",
            ]
        ]
        dict_methods_od_files["kddcup-guess_passwd_vs_satan"] = [
            el
            for el in dict_methods_od_files["kddcup-land_vs_satan"]
            if el
            not in [
                "LOP_basic",
                "DBSCAN_02",
                "DBSCAN_04",
                "DBSCAN_05",
                "DBSCAN_06",
                "DBSCAN_08",
                "DBSCAN_09",
                "DBSCAN_10",
            ]
        ]
        dict_methods_od_files["mammography"] = [
            el for el in dict_methods_od_files["mammography"] if ("PYOD_COF" not in el)
        ]
        dict_methods_od_files['winequality-white-3_vs_7'] = [
            el for el in dict_methods_od_files["winequality-white-3_vs_7"] if ("LOP_basic" not in el)
        ]
        dict_methods_od_files['kddcup-buffer_overflow_vs_back'] = [
            el for el in dict_methods_od_files["kddcup-buffer_overflow_vs_back"] if ("LOP_basic" not in el)
        ]
        dict_methods_od_files['kddcup-land_vs_portsweep'] = [
            el for el in dict_methods_od_files['kddcup-land_vs_portsweep'] if ("LOP_basic" not in el)
        ]
        dict_methods_od_files['kddcup-rootkit-imap_vs_back'] = [
            el for el in dict_methods_od_files['kddcup-rootkit-imap_vs_back'] if ("LOP_basic" not in el)
        ]
        dict_methods_od_files['coil_2000'] = [
            el for el in dict_methods_od_files['coil_2000'] if ("LOP_basic" not in el) and ("DBSCAN" not in el)
        ]
        dict_methods_od_files['page-blocks-1-3_vs_4'] = [
            el for el in dict_methods_od_files['page-blocks-1-3_vs_4'] if ("LOP_basic" not in el)
        ]
    elif method == 'OD-BINOD^RED':
        for k, v in dict_methods_od_files.items():
            dict_methods_od_files[k] = [
                el
                for el in dict_methods_od_files[k]
                if el
                not in [
                    "LOP_basic"
                ]
            ]
    return dict_methods_od_files


def get_dict_methods_red(data_fold="data", method_list='default'):
    filenames = os.listdir(data_fold)
    if method_list in ['default','xgbod']:
        dict_methods_red_files = {
            filename: list_methods_red.copy() for filename in filenames
        }
    elif method_list == 'light':
        dict_methods_red_files = {
            filename: list_methods_red_light.copy() for filename in filenames
        }
    return dict_methods_red_files


def get_bootstrap_estimates(yy, iters=100, meta="FROID", classifier_string="lgbm"):
    results = pd.DataFrame()
    for b in range(iters + 1):
        if b == 0:
            db = yy.copy()
        else:
            db = yy.sample(len(yy), random_state=b, replace=True)
            sum_elements = db.true.sum()
            time = 0
            possible_singles = np.unique(yy.true) * len(yy.true)
            elements = [int(el) for el in possible_singles]
            while sum_elements in elements:
                db = yy.sample(len(yy), random_state=b + iters * time, replace=True)
                time += 1
                sum_elements = db.true.sum()
        db = db.reset_index()
        metrics = get_results_binary(db["true"], db["scores"], db["preds"])
        tmp = pd.DataFrame(
            [metrics],
            columns=[
                "auc",
                "gini",
                "accuracy",
                "precision",
                "recall",
                "sensitivity_score",
                "specificity_score",
                "geom_mean_score",
                "average_precision_score",
                "f1-score",
            ],
        )
        tmp["boot_iter"] = b
        results = pd.concat([results, tmp], axis=0)
    return results

def get_results_binary(y_true, y_score, y_preds):
    auc = skm.roc_auc_score(y_true, y_score)
    gini = (2 * auc) - 1
    acc = skm.accuracy_score(y_true, y_preds)
    pre = skm.precision_score(y_true, y_preds)
    rec = skm.recall_score(y_true, y_preds)
    ses = imm.sensitivity_score(y_true, y_preds, average="binary")
    sps = imm.specificity_score(y_true, y_preds, average="binary")
    gms = imm.geometric_mean_score(y_true, y_preds, average="binary")
    aps = skm.average_precision_score(y_true, y_score)
    f1s = skm.f1_score(y_true, y_preds, average="binary")
    return [auc, gini, acc, pre, rec, ses, sps, gms, aps, f1s]

def get_dictionaries_methods(ml='default', data_fold='data'):
    dict_method_red = {}
    dict_method_od = {}
    print("get_dictionaries_methods ml == {}".format(ml))
    odbinodred_od_files =  get_dict_methods_od(data_fold, method='OD-BINOD-RED', method_list=ml)
    red_files = get_dict_methods_red(data_fold, method_list=ml)
    odod_od_files = get_dict_methods_od(data_fold, method='OD-BINOD^OD-BINOD', method_list=ml)
    redod_od_files = get_dict_methods_od(data_fold, method='RED^OD-BINOD', method_list=ml)
    filelist = os.listdir(data_fold)
    for f in filelist:
            tmp1 = {'OD-BINOD-RED': odbinodred_od_files[f].copy(),
                    'OD-BINOD^OD-BINOD': odod_od_files[f].copy(),
                    'RED^OD-BINOD': redod_od_files[f].copy()}
            tmp2 = {'OD-BINOD-RED': red_files[f].copy(),
                    'OD-BINOD^RED': red_files[f].copy(),
                    'RED^RED': red_files[f].copy()}
            tmp1['RED^OD-BINOD'] = [el for el in tmp1['RED^OD-BINOD'] if ("LOP_basic" not in el)].copy()
            dict_method_od[f] = tmp1.copy()
            dict_method_red[f] = tmp2.copy()
    try:
        dict_method_red['kddcup-land_vs_portsweep']['OD-BINOD^RED'] = [el for el in
                                                                  dict_method_red['kddcup-land_vs_portsweep']['RED^RED']
                                                                  if (el not in ['TSNE', 'ISO', 'LLE_HESS', 'SE'])]
    except:
        print("todo bueno")
    return dict_method_od, dict_method_red

def check_columns(df_train, df_test, features):
    l1 = [col for col in df_train.columns.tolist() if col not in ['TARGET', 'INDEX']].copy()
    l2 = [col for col in df_test.columns.tolist() if col not in ['TARGET', 'INDEX']].copy()
    x1 = set(l1)
    x2 = set(l2)
    x3 = (x1 - x2).copy()
    feats_train_not_in_test = list(x3)
    print(feats_train_not_in_test)
    features = [
        col
        for col in features.copy()
        if (col not in feats_train_not_in_test)
    ].copy()
    inf_mean_cols = (
        df_train[features]
        .columns[np.where(np.mean(df_train[features]) == np.inf)]
        .tolist()
    )
    inf_var_cols = (
        df_train[features]
        .columns[np.where(np.var(df_train[features]) == np.inf)]
        .tolist()
    )
    inf_mean_cols_test = (
        df_test[features]
        .columns[np.where(np.mean(df_test[features]) == np.inf)]
        .tolist()
    )
    inf_var_cols_test = (
        df_test[features]
        .columns[np.where(np.var(df_test[features]) == np.inf)]
        .tolist()
    )
    features = [
        col
        for col in features
        if (col not in inf_mean_cols)
           & (col not in inf_mean_cols_test)
           & (col not in inf_var_cols)
           & (col not in inf_var_cols_test)
    ]
    return features

def check_prescaler_df(df):
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df.dropna()
    inf_mean_cols = (
        df.columns[np.where(np.mean(df) == np.inf)]
        .tolist()
    )
    inf_var_cols = (
        df.columns[np.where(np.var(df) == np.inf)]
        .tolist()
    )
    features = [
        col
        for col in df.columns
        if (col not in inf_mean_cols)
           & (col not in inf_var_cols)
    ]
    return df[features].copy()

def get_whole_FROID(X_train, X_test, y_train, y_test, dict_method_od, dict_method_red,
                    index_train=None, index_test=None, filename='stars', data_fold = 'data', meta='FROID'):
        random.seed(42)
        np.random.seed(42)
        od_binod_red_path_train = '{}/{}/{}_{}_OD-BINOD-RED_train.csv'.format(data_fold, filename, filename, meta)
        odod_path_train = '{}/{}/{}_{}_OD-BINOD^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta)
        redod_path_train = '{}/{}/{}_{}_RED^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta)
        redred_path_train = '{}/{}/{}_{}_RED^RED_train.csv'.format(data_fold, filename, filename, meta)
        odred_path_train = '{}/{}/{}_{}_OD-BINOD^RED_train.csv'.format(data_fold, filename, filename, meta)
        od_binod_red_path_test = '{}/{}/{}_{}_OD-BINOD-RED_test.csv'.format(data_fold, filename, filename, meta)
        odod_path_test = '{}/{}/{}_{}_OD-BINOD^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta)
        redod_path_test = '{}/{}/{}_{}_RED^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta)
        redred_path_test = '{}/{}/{}_{}_RED^RED_test.csv'.format(data_fold, filename, filename, meta)
        odred_path_test = '{}/{}/{}_{}_OD-BINOD^RED_test.csv'.format(data_fold, filename, filename, meta)
        s1 = FROID('OD-BINOD-RED', list_methods_od=dict_method_od[filename]['OD-BINOD-RED'],
                   list_methods_red=dict_method_red[filename]['OD-BINOD-RED'])
        s2_1 = FROID('OD-BINOD', list_methods_od=dict_method_od[filename]['OD-BINOD^OD-BINOD'], list_methods_red=[],
                     inception=True, previous_stage='OD-BINOD')
        s2_2 = FROID('OD-BINOD', list_methods_od=dict_method_od[filename]['RED^OD-BINOD'], list_methods_red=[],
                     inception=True, previous_stage='RED')
        s2_3 = FROID('RED', list_methods_od=[], list_methods_red=dict_method_red[filename]['RED^RED'], inception=True,
                     previous_stage='RED')
        s2_4 = FROID('RED', list_methods_od=[], list_methods_red=dict_method_red[filename]['OD-BINOD^RED'],
                     inception=True, previous_stage='OD-BINOD')
        atts_level0 = X_train.columns.tolist()
        st = time.time()
        time_df = pd.DataFrame(columns=['step', 'time'])
        if (os.path.exists(od_binod_red_path_test)) & (os.path.exists(od_binod_red_path_train)):
            print("path exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            df_train_s1 = pd.read_csv(od_binod_red_path_train)
            df_test_s1 = pd.read_csv(od_binod_red_path_test)
            time_df = pd.read_csv('{}/{}/{}_{}_time_OD-BINOD-RED.csv'.format(data_fold, filename, filename, meta))
            atts_level0 = X_train.columns.tolist()
            atts_s1 = [col for col in df_train_s1.columns if (col not in ['TARGET']) & (col not in atts_level0)]
            X_train_s1 = df_train_s1[atts_s1].copy()
            X_test_s1 = df_test_s1[atts_s1].copy()
        else:
            #here we fit the first level for FROID
            s1.fit(X_train, y_train)
            et = time.time()
            time_df = pd.DataFrame([['OD-BINOD-RED', s1.time_to_expand]], columns = ['step','time'])
            #here we save time to expand first level
            time_df.to_csv('{}/{}/{}_{}_time_OD-BINOD-RED.csv'.format(data_fold, filename, filename, meta), index=False)
            #here we save level 1 (od-binod-red) on training and test
            X_train_s1 = s1.data_train.copy().dropna(axis=1)
            X_test_s1 = s1.transform(X_test).copy().dropna(axis=1)
            #here we start elaborating scaling
            scaler_s1 = RobustScaler()
            # here we check there are no missing values cols for scaling
            X_train_s1 = check_prescaler_df(X_train_s1)
            X_test_s1 = check_prescaler_df(X_test_s1)
            atts_s1_train = [col for col in X_train_s1.columns if (col not in atts_level0) &
                             (col not in ['INDEX','TARGET'])]
            atts_s1_test = [col for col in X_test_s1.columns if (col not in atts_level0) &
                             (col not in ['INDEX','TARGET'])]
            atts_s1 = sorted(list(set(atts_s1_train.copy()).intersection(set(atts_s1_test.copy()))), reverse=True)
            #here we define columns to be scaled (i.e. those that are not binary)
            atts_s1_toscale = [el for el in atts_s1 if ('bin_score' not in el)]
            #here we fit the scaler
            scaler_s1.fit(X_train_s1[atts_s1_toscale])
            #here we predict values scaled
            X_test_s1[atts_s1_toscale] = scaler_s1.transform(X_test_s1[atts_s1_toscale])
            X_train_s1[atts_s1_toscale] = scaler_s1.transform(X_train_s1[atts_s1_toscale])
            #here we save the scaler
            scaler_path = "{}/{}/{}_scaler_{}_OD-BINOD-RED.save".format(data_fold, filename, filename, meta)
            scaler_path_pickle = "{}/{}/{}_scaler_{}_OD-BINOD-RED.pkl".format(data_fold, filename, filename, meta)
            joblib.dump(scaler_s1, scaler_path)
            joblib.dump(scaler_s1, scaler_path_pickle)
            #here we write the data
            df_train_s1 = pd.concat([X_train_s1[atts_s1], y_train], axis=1)
            df_train_s1.to_csv('{}/{}/{}_{}_OD-BINOD-RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
            df_test_s1 = pd.concat([X_test_s1[atts_s1], y_test], axis=1)
            df_test_s1.to_csv('{}/{}/{}_{}_OD-BINOD-RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
        #here we set variables that are used in inception step
        atts_s2_od_train = [col for col in X_train_s1.columns.tolist() if ('scores_' in col) & (col not in atts_level0)]
        atts_s2_red_train = [col for col in X_train_s1.columns.tolist() if ('comp_' in col) & (col not in atts_level0)]
        atts_s2_od_test = [col for col in X_test_s1.columns.tolist() if ('scores_' in col) & (col not in atts_level0)]
        atts_s2_red_test = [col for col in X_test_s1.columns.tolist() if ('comp_' in col) & (col not in atts_level0)]
        atts_s2_od = sorted(list(set(atts_s2_od_train.copy()).intersection(set(atts_s2_od_test.copy()))), reverse=True)
        atts_s2_red = sorted(list(set(atts_s2_red_train.copy()).intersection(set(atts_s2_red_test.copy()))), reverse=True)
        atts_level1 = atts_s2_od.copy() + atts_s2_red.copy()
        #here we fit INCEPTION FROID over scores
        # OD-BINOD^OD-BINOD
        if (os.path.exists(odod_path_test)) & (os.path.exists(odod_path_train)):
            df_train_s2_1 = pd.read_csv(odod_path_train)
            df_test_s2_1 = pd.read_csv(odod_path_test)
            atts_s2_1 = [col for col in df_train_s2_1.columns if (col not in ['TARGET']) & (col not in atts_level0)]
            try:
                X_train_s2_1 = df_train_s2_1[atts_s2_1].copy()
            except:
                import pdb; pdb.set_trace()
            X_test_s2_1 = df_test_s2_1[atts_s2_1].copy()
            # time_df = pd.read_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
        else:
            try:
                s2_1.fit(X_train_s1[atts_s2_od], y_train)
            except:
                import pdb; pdb.set_trace()
            tm = pd.DataFrame([['OD-BINOD^OD-BINOD', s2_1.time_to_expand]], columns = ['step','time'])
            time_df = pd.concat([time_df, tm], axis=0)
            time_df.to_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta), index=False)
            X_train_s2_1 = s2_1.data_train.copy().dropna(axis=1)
            X_test_s2_1 = s2_1.transform(X_test_s1[atts_s2_od]).copy().dropna(axis=1)
            X_train_s2_1 = check_prescaler_df(X_train_s2_1)
            X_test_s2_1 = check_prescaler_df(X_test_s2_1)
            scaler_s2_1 = RobustScaler()
            atts_s2_1_train = [col for col in X_train_s2_1.columns if (col not in atts_s2_od) &
                             (col not in ['INDEX', 'TARGET'])]
            atts_s2_1_test = [col for col in X_test_s2_1.columns if (col not in atts_s2_od) &
                            (col not in ['INDEX', 'TARGET'])]
            atts_s2_1 = sorted(list(set(atts_s2_1_train.copy()).intersection(set(atts_s2_1_test.copy()))), reverse=True)
            atts_s2_1_toscale = [el for el in atts_s2_1 if ('bin_score' not in el)]
            scaler_s2_1.fit(X_train_s2_1[atts_s2_1_toscale])
            scaler_path = "{}/{}/{}_scaler_{}_OD-BINOD^OD-BINOD.save".format(data_fold, filename, filename, meta)
            scaler_path_pickle = "{}/{}/{}_scaler_{}_OD-BINOD^OD-BINOD.pkl".format(data_fold, filename, filename, meta)
            joblib.dump(scaler_s2_1, scaler_path)
            joblib.dump(scaler_s2_1, scaler_path_pickle)
            X_train_s2_1[atts_s2_1_toscale] = scaler_s2_1.transform(X_train_s2_1[atts_s2_1_toscale]).copy()
            X_test_s2_1[atts_s2_1_toscale] = scaler_s2_1.transform(X_test_s2_1[atts_s2_1_toscale]).copy()
            df_train_s2_1 = pd.concat([X_train_s2_1, y_train], axis=1)
            df_test_s2_1 = pd.concat([X_test_s2_1, y_test], axis=1)
            df_train_s2_1.to_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta),
                                 index=False)

            df_test_s2_1.to_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta), index=False)
        # RED^OD-BINOD
        if (os.path.exists(redod_path_test)) & (os.path.exists(redod_path_train)):
            df_train_s2_2 = pd.read_csv(redod_path_train)
            df_test_s2_2 = pd.read_csv(redod_path_test)
            atts_s2_2 = [col for col in df_train_s2_2.columns if (col not in ['TARGET']) & (col not in atts_level0)]
            X_train_s2_2 = df_train_s2_2[atts_s2_2].copy()
            X_test_s2_2 = df_test_s2_2[atts_s2_2].copy()
            # time_df = pd.read_csv('{}/{}/{}_{}_RED^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
        else:
            #here we fit RED^OD-BINOD
            s2_2.fit(X_train_s1[atts_s2_red], y_train)
            tm = pd.DataFrame([['RED^OD-BINOD', s2_2.time_to_expand]], columns=['step', 'time'])
            time_df = pd.concat([time_df, tm], axis=0)
            time_df.to_csv('{}/{}/{}_{}_RED^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta), index=False)
            X_train_s2_2 = s2_2.data_train.copy().dropna(axis=1)
            X_test_s2_2 = s2_2.transform(X_test_s1[atts_s2_red]).copy().dropna(axis=1)
            X_train_s2_2 = check_prescaler_df(X_train_s2_2)
            X_test_s2_2 = check_prescaler_df(X_test_s2_2)
            scaler_s2_2 = RobustScaler()
            atts_s2_2_train = [col for col in X_train_s2_2.columns if (col not in atts_s2_od) &
                               (col not in ['INDEX', 'TARGET'])]
            atts_s2_2_test = [col for col in X_test_s2_2.columns if (col not in atts_s2_od) &
                              (col not in ['INDEX', 'TARGET'])]
            atts_s2_2 = sorted(list(set(atts_s2_2_train.copy()).intersection(set(atts_s2_2_test.copy()))), reverse=True)
            atts_s2_2_toscale = [el for el in atts_s2_2 if ('bin_score' not in el)]
            scaler_s2_2.fit(X_train_s2_2[atts_s2_2_toscale])
            scaler_path = "{}/{}/{}_scaler_{}_RED^OD-BINOD.save".format(data_fold, filename, filename, meta)
            scaler_path_pickle = "{}/{}/{}_scaler_{}_RED^OD-BINOD.pkl".format(data_fold, filename, filename, meta)
            joblib.dump(scaler_s2_2, scaler_path)
            joblib.dump(scaler_s2_2, scaler_path_pickle)
            X_train_s2_2[atts_s2_2_toscale] = scaler_s2_2.transform(X_train_s2_2[atts_s2_2_toscale]).copy()
            X_test_s2_2[atts_s2_2_toscale] = scaler_s2_2.transform(X_test_s2_2[atts_s2_2_toscale]).copy()
            df_train_s2_2 = pd.concat([X_train_s2_2, y_train], axis=1)
            df_test_s2_2 = pd.concat([X_test_s2_2, y_test], axis=1)
            df_train_s2_2.to_csv('{}/{}/{}_{}_RED^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta), index=False)
            df_test_s2_2.to_csv('{}/{}/{}_{}_RED^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta), index=False)
        #RED^RED
        if (os.path.exists(redred_path_test)) & (os.path.exists(redred_path_train)):
            df_train_s2_3 = pd.read_csv(redred_path_train)
            df_test_s2_3 = pd.read_csv(redred_path_test)
            atts_s2_3 = [col for col in df_train_s2_3.columns if (col not in ['TARGET']) & (col not in atts_level0)]
            X_train_s2_3 = df_train_s2_3[atts_s2_3].copy()
            X_test_s2_3 = df_test_s2_3[atts_s2_3].copy()
            # time_df = pd.read_csv('{}/{}/{}_{}_RED^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
        else:
            s2_3.fit(X_train_s1[atts_s2_red], y_train)
            tm = pd.DataFrame([['RED^RED', s2_3.time_to_expand]], columns=['step', 'time'])
            time_df = pd.concat([time_df, tm], axis=0)
            time_df.to_csv('{}/{}/{}_{}_RED^RED_time.csv'.format(data_fold, filename, filename, meta), index=False)
            X_train_s2_3 = s2_3.data_train.copy()
            X_test_s2_3 = s2_3.transform(X_test_s1[atts_s2_red])
            X_train_s2_3 = check_prescaler_df(X_train_s2_3)
            X_test_s2_3 = check_prescaler_df(X_test_s2_3)
            scaler_s2_3 = RobustScaler()
            atts_s2_3_train = [col for col in X_train_s2_3.columns if (col not in atts_s2_od) &
                               (col not in ['INDEX', 'TARGET'])]
            atts_s2_3_test = [col for col in X_test_s2_3.columns if (col not in atts_s2_od) &
                              (col not in ['INDEX', 'TARGET'])]
            atts_s2_3 = sorted(list(set(atts_s2_3_train.copy()).intersection(set(atts_s2_3_test.copy()))), reverse=True)
            atts_s2_3_toscale = [el for el in atts_s2_3 if ('bin_score' not in el)]
            scaler_s2_3.fit(X_train_s2_3[atts_s2_3_toscale])
            scaler_path = "{}/{}/{}_scaler_{}_RED^RED.save".format(data_fold, filename, filename, meta)
            scaler_path_pickle = "{}/{}/{}_scaler_{}_RED^RED.pkl".format(data_fold, filename, filename, meta)
            joblib.dump(scaler_s2_3, scaler_path)
            joblib.dump(scaler_s2_3, scaler_path_pickle)
            X_train_s2_3[atts_s2_3_toscale] = scaler_s2_3.transform(X_train_s2_3[atts_s2_3_toscale]).copy()
            X_test_s2_3[atts_s2_3_toscale] = scaler_s2_3.transform(X_test_s2_3[atts_s2_3_toscale]).copy()
            df_train_s2_3 = pd.concat([X_train_s2_3, y_train], axis=1)
            df_test_s2_3 = pd.concat([X_test_s2_3, y_test], axis=1)
            df_train_s2_3.to_csv('{}/{}/{}_{}_RED^RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
            df_test_s2_3.to_csv('{}/{}/{}_{}_RED^RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
        #OD-BINOD^RED
        if (os.path.exists(odred_path_test)) & (os.path.exists(odred_path_train)):
            df_train_s2_4 = pd.read_csv(odred_path_train)
            df_test_s2_4 = pd.read_csv(odred_path_test)
            atts_s2_4 = [col for col in df_train_s2_4.columns if (col not in ['TARGET']) & (col not in atts_level0)]
            X_train_s2_4 = df_train_s2_4[atts_s2_4].copy()
            X_test_s2_4 = df_test_s2_4[atts_s2_4].copy()
            # time_df = pd.read_csv('{}/{}/{}_{}_RED^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
        else:
            s2_4.fit(X_train_s1[atts_s2_od], y_train)
            tm = pd.DataFrame([['OD-BINOD^RED', s2_4.time_to_expand]], columns=['step', 'time'])
            time_df = pd.concat([time_df, tm], axis=0)
            time_df.to_csv('{}/{}/{}_{}_OD-BINOD^RED_time.csv'.format(data_fold, filename, filename, meta), index=False)
            X_train_s2_4 = s2_4.data_train
            X_test_s2_4 = s2_4.transform(X_test_s1[atts_s2_od])
            X_train_s2_4 = check_prescaler_df(X_train_s2_4)
            X_test_s2_4 = check_prescaler_df(X_test_s2_4)
            scaler_s2_4 = RobustScaler()
            atts_s2_4_train = [col for col in X_train_s2_4.columns if (col not in atts_s2_od) &
                               (col not in ['INDEX', 'TARGET'])]
            atts_s2_4_test = [col for col in X_test_s2_4.columns if (col not in atts_s2_od) &
                              (col not in ['INDEX', 'TARGET'])]
            atts_s2_4 = sorted(list(set(atts_s2_4_train.copy()).intersection(set(atts_s2_4_test.copy()))), reverse=True)
            atts_s2_4_toscale = [el for el in atts_s2_4 if ('bin_score' not in el)]
            scaler_s2_4.fit(X_train_s2_4[atts_s2_4_toscale])
            scaler_path = "{}/{}/{}_scaler_{}_OD-BINOD^RED.save".format(data_fold, filename, filename, meta)
            scaler_path_pickle = "{}/{}/{}_scaler_{}_OD-BINOD^RED.pkl".format(data_fold, filename, filename, meta)
            joblib.dump(scaler_s2_4, scaler_path)
            joblib.dump(scaler_s2_4, scaler_path_pickle)
            X_train_s2_4[atts_s2_4_toscale] = scaler_s2_4.transform(X_train_s2_4[atts_s2_4_toscale]).copy()
            X_test_s2_4[atts_s2_4_toscale] = scaler_s2_4.transform(X_test_s2_4[atts_s2_4_toscale]).copy()
            df_train_s2_4 = pd.concat([X_train_s2_4, y_train], axis=1)
            df_test_s2_4 = pd.concat([X_test_s2_4, y_test], axis=1)
            df_train_s2_4.to_csv('{}/{}/{}_{}_OD-BINOD^RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
            df_test_s2_4.to_csv('{}/{}/{}_{}_OD-BINOD^RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
        #WHOLE DATASET TRAINING
        atts_odod_train = [col for col in X_train_s2_1.columns.tolist() if (col not in atts_level1)].copy()
        atts_redod_train = [col for col in X_train_s2_2.columns.tolist() if (col not in atts_level1)].copy()
        atts_redred_train = [col for col in X_train_s2_3.columns.tolist() if (col not in atts_level1)].copy()
        atts_odred_train = [col for col in X_train_s2_4.columns.tolist() if (col not in atts_level1)].copy()
        if index_train is None:
            db_train = pd.concat([X_train, X_train_s1[atts_level1], X_train_s2_1[atts_odod_train],
                        X_train_s2_2[atts_redod_train], X_train_s2_3[atts_redred_train], X_train_s2_4[atts_odred_train], y_train], axis=1)
        else:
            db_train = pd.concat([index_train, X_train, X_train_s1[atts_level1], X_train_s2_1[atts_odod_train],
                        X_train_s2_2[atts_redod_train], X_train_s2_3[atts_redred_train], X_train_s2_4[atts_odred_train], y_train], axis=1)

        db_train.to_csv('{}/{}/{}_{}_WHOLE_train.csv'.format(data_fold, filename, filename, meta), index=False)
        #WHOLE DATASET TEST
        atts_odod_test = [col for col in X_test_s2_1.columns.tolist() if (col not in atts_level1)].copy()
        atts_redod_test = [col for col in X_test_s2_2.columns.tolist() if (col not in atts_level1)].copy()
        atts_redred_test = [col for col in X_test_s2_3.columns.tolist() if (col not in atts_level1)].copy()
        atts_odred_test = [col for col in X_test_s2_4.columns.tolist() if (col not in atts_level1)].copy()
        if index_test is None:
            db_test = pd.concat([X_test, X_test_s1[atts_level1], X_test_s2_1[atts_odod_test],
                        X_test_s2_2[atts_redod_test], X_test_s2_3[atts_redred_test], X_test_s2_4[atts_odred_test], y_test], axis=1)
        else:
            db_test = pd.concat([index_test, X_test, X_test_s1[atts_level1], X_test_s2_1[atts_odod_test],
                        X_test_s2_2[atts_redod_test], X_test_s2_3[atts_redred_test], X_test_s2_4[atts_odred_test], y_test], axis=1)
        db_test.to_csv('{}/{}/{}_{}_WHOLE_test.csv'.format(data_fold, filename, filename, meta), index=False)
        time_s1 = s1.time_to_expand
        time_s2_1 = s2_1.time_to_expand
        time_s2_2 = s2_2.time_to_expand
        time_s2_3 = s2_3.time_to_expand
        time_s2_4 = s2_4.time_to_expand
        time_whole = time_s1+time_s2_1+time_s2_2+time_s2_3+time_s2_4
        list_times = list(zip(['OD-BINOD-RED', 'OD-BINOD^OD-BINOD', 'RED^OD-BINOD', 'RED^RED', 'OD-BINOD^RED', 'WHOLE'],
                              [time_s1, time_s2_1, time_s2_2, time_s2_3, time_s2_4, time_whole]))
        time_df = pd.DataFrame(list_times, columns = ['FROID_method', 'time'])
        if os.path.exists('{}/{}/{}_{}_WHOLE_time.csv'.format(data_fold, filename, filename, meta)) == False:
            time_df.to_csv('{}/{}/{}_{}_WHOLE_time.csv'.format(data_fold, filename, filename, meta), index=False)
        return db_train, db_test


def get_whole_FROID_FULL(X_train, X_test, y_train, y_test, dict_method_od, dict_method_red,
                    index_train=None, index_test=None, filename='stars', data_fold='data', meta='FROIDFULL'):
    #check no method with y (i.e. LDA) is within dictionaries:
    np.random.seed(42)
    random.seed(42)
    try:
        for k,v in dict_method_red[filename].items():
            v = [el for el in v if ('LDA' not in el)]
    except: import pdb; pdb.set_trace()
    od_binod_red_path_train = '{}/{}/{}_{}_OD-BINOD-RED_train.csv'.format(data_fold, filename, filename, meta)
    odod_path_train = '{}/{}/{}_{}_OD-BINOD^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta)
    redod_path_train = '{}/{}/{}_{}_RED^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta)
    redred_path_train = '{}/{}/{}_{}_RED^RED_train.csv'.format(data_fold, filename, filename, meta)
    odred_path_train = '{}/{}/{}_{}_OD-BINOD^RED_train.csv'.format(data_fold, filename, filename, meta)
    od_binod_red_path_test = '{}/{}/{}_{}_OD-BINOD-RED_test.csv'.format(data_fold, filename, filename, meta)
    odod_path_test = '{}/{}/{}_{}_OD-BINOD^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta)
    redod_path_test = '{}/{}/{}_{}_RED^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta)
    redred_path_test = '{}/{}/{}_{}_RED^RED_test.csv'.format(data_fold, filename, filename, meta)
    odred_path_test = '{}/{}/{}_{}_OD-BINOD^RED_test.csv'.format(data_fold, filename, filename, meta)
    #here we prepare all the various FROID for different inceptions
    s1 = FROID('OD-BINOD-RED', list_methods_od=dict_method_od[filename]['OD-BINOD-RED'],
               list_methods_red=dict_method_red[filename]['OD-BINOD-RED'])
    s2_1 = FROID('OD-BINOD', list_methods_od=dict_method_od[filename]['OD-BINOD^OD-BINOD'], list_methods_red=[],
                 inception=True, previous_stage='OD-BINOD')
    s2_2 = FROID('OD-BINOD', list_methods_od=dict_method_od[filename]['RED^OD-BINOD'], list_methods_red=[],
                 inception=True, previous_stage='RED')
    s2_3 = FROID('RED', list_methods_od=[], list_methods_red=dict_method_red[filename]['RED^RED'], inception=True,
                 previous_stage='RED')
    s2_4 = FROID('RED', list_methods_od=[], list_methods_red=dict_method_red[filename]['OD-BINOD^RED'], inception=True,
                 previous_stage='OD-BINOD')
    #here we store the columns of original dataset
    atts_level0 = X_train.columns.tolist()
    #here we set the limit of training set
    final_pos = X_train.shape[0]
    #here we append train and test set to make it a full sample
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    #here we reset the index, as dataframe would have double indexes otherwise
    X_full.reset_index(drop=True, inplace=True)
    X_full.reset_index(drop=True, inplace=True)
    time_df = pd.DataFrame()
    if (os.path.exists(od_binod_red_path_test)) & (os.path.exists(od_binod_red_path_train)):
        print("path exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        df_train_s1 = pd.read_csv(od_binod_red_path_train)
        df_test_s1 = pd.read_csv(od_binod_red_path_test)
        atts_level0 = X_train.columns.tolist()
        atts_s1 = [col for col in df_train_s1.columns if (col not in ['TARGET']) & (col not in atts_level0)]
        X_train_s1 = df_train_s1[atts_s1].copy()
        X_test_s1 = df_test_s1[atts_s1].copy()
        X_full_s1 = pd.concat([X_train_s1[atts_s1], X_test_s1[atts_s1]], axis=0)
    else:
        #here we fit FROID on the full dataset
        s1.fit(X_full, y_full)
        tmp = s1.data_train.copy().dropna(axis=1)
        #here we store the train set and the test set
        X_train_s1 = tmp.iloc[:final_pos]
        X_test_s1 = tmp.iloc[final_pos:]
        X_test_s1.reset_index(drop=True, inplace=True)
        #here we scale newly created columns using Robust Scaler
        scaler_s1 = RobustScaler()
        #here we check we use variables existing in both training and test
        atts_s1_train = [col for col in X_train_s1.columns if (col not in atts_level0) &
                         (col not in ['INDEX', 'TARGET'])]
        atts_s1_test = [col for col in X_test_s1.columns if (col not in atts_level0) &
                        (col not in ['INDEX', 'TARGET'])]
        atts_s1 = sorted(list(set(atts_s1_train.copy()).intersection(set(atts_s1_test.copy()))), reverse=True)
        #here we fit the scaler on training instances
        scaler_s1.fit(X_train_s1[atts_s1])
        #here we scale the data
        X_test_s1[atts_s1] = scaler_s1.transform(X_test_s1[atts_s1])
        X_train_s1[atts_s1] = scaler_s1.transform(X_train_s1[atts_s1])
        X_full_s1 = pd.concat([X_train_s1[atts_s1], X_test_s1[atts_s1]], axis=0)
        #here we concat and we write both train and test
        df_train_s1 = pd.concat([X_train_s1[atts_s1], y_train], axis=1)
        df_train_s1.to_csv('{}/{}/{}_{}_OD-BINOD-RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
        df_test_s1 = pd.concat([X_test_s1[atts_s1], y_test], axis=1)
        df_test_s1.to_csv('{}/{}/{}_{}_OD-BINOD-RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
    #here we define variables for inception: atts_s2_od are all columns with scores (i.e. OD methods) ,
    #atts_s2_red are all columns created by FR methods
    atts_s2_od = [col for col in X_full_s1.columns if ('scores_' in col) & (col not in atts_level0)]
    atts_s2_red = [col for col in X_full_s1.columns if ('comp_' in col) & (col not in atts_level0)]
    #we add together these two levels as atts_level1
    atts_level1 = atts_s2_od.copy() + atts_s2_red.copy()
    # here we fit INCEPTION FROID over scores
    # OD-BINOD^OD-BINOD
    # here we fit a FROID over all the dataset using only OD methods over OD columns previously created
    print(X_full_s1[atts_s2_od].head().iloc[:10])
    if (os.path.exists(odod_path_test)) & (os.path.exists(odod_path_train)):
        df_train_s2_1 = pd.read_csv(odod_path_train)
        df_test_s2_1 = pd.read_csv(odod_path_test)
        atts_s2_1 = [col for col in df_train_s2_1.columns if (col not in ['TARGET']) & (col not in atts_level0)]
        try:
            X_train_s2_1 = df_train_s2_1[atts_s2_1].copy()
        except:
            import pdb;
            pdb.set_trace()
        X_test_s2_1 = df_test_s2_1[atts_s2_1].copy()
        # time_df = pd.read_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
    else:
        try:
            s2_1.fit(X_full_s1[atts_s2_od], y_full)
        except: import pdb; pdb.set_trace()
        X_train_s2_1 = s2_1.data_train.iloc[:final_pos]
        X_test_s2_1 = s2_1.data_train.iloc[final_pos:]
        X_test_s2_1.reset_index(drop=True, inplace=True)
        scaler_s2_1 = RobustScaler()
        atts_s2_1_train = [col for col in X_train_s2_1.columns if (col not in atts_s2_od) &
                           (col not in ['INDEX', 'TARGET'])]
        atts_s2_1_test = [col for col in X_test_s2_1.columns if (col not in atts_s2_od) &
                          (col not in ['INDEX', 'TARGET'])]
        atts_s2_1 = sorted(list(set(atts_s2_1_train.copy()).intersection(set(atts_s2_1_test.copy()))), reverse=True)
        atts_s2_1_toscale = [el for el in atts_s2_1 if ('bin_score' not in el)]
        try:
            scaler_s2_1.fit(X_train_s2_1[atts_s2_1_toscale])
        except:
            import pdb; pdb.set_trace()
        try:
            X_train_s2_1[atts_s2_1_toscale] = scaler_s2_1.transform(X_train_s2_1[atts_s2_1_toscale]).copy()
            X_test_s2_1[atts_s2_1_toscale] = scaler_s2_1.transform(X_test_s2_1[atts_s2_1_toscale]).copy()
        except:
            import pdb
            pdb.set_trace()
        df_train_s2_1 = pd.concat([X_train_s2_1, y_train], axis=1)
        df_test_s2_1 = pd.concat([X_test_s2_1, y_test], axis=1)
        df_train_s2_1.to_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta),
                             index=False)

        df_test_s2_1.to_csv('{}/{}/{}_{}_OD-BINOD^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta), index=False)
    # RED^OD-BINOD
    if (os.path.exists(redod_path_test)) & (os.path.exists(redod_path_train)):
        df_train_s2_2 = pd.read_csv(redod_path_train)
        df_test_s2_2 = pd.read_csv(redod_path_test)
        atts_s2_2 = [col for col in df_train_s2_2.columns if (col not in ['TARGET', 'INDEX']) & (col not in atts_level0)]
        X_train_s2_2 = df_train_s2_2[atts_s2_2].copy()
        X_test_s2_2 = df_test_s2_2[atts_s2_2].copy()
    else:
        s2_2.fit(X_full_s1[atts_s2_red], y_full)
        X_train_s2_2 = s2_2.data_train.iloc[:final_pos].copy()
        X_test_s2_2 = s2_2.data_train.iloc[final_pos:].copy()
        X_test_s2_2.reset_index(drop=True, inplace=True)
        scaler_s2_2 = RobustScaler()
        atts_s2_2_train = [col for col in X_train_s2_2.columns if (col not in atts_s2_red) &
                           (col not in ['INDEX', 'TARGET'])]
        atts_s2_2_test = [col for col in X_test_s2_2.columns if (col not in atts_s2_red) &
                          (col not in ['INDEX', 'TARGET'])]
        atts_s2_2 = sorted(list(set(atts_s2_2_train.copy()).intersection(set(atts_s2_2_test.copy()))), reverse=True)
        atts_s2_2_toscale = [el for el in atts_s2_2 if ('bin_score' not in el)]
        scaler_s2_2.fit(X_train_s2_2[atts_s2_2_toscale])
        X_train_s2_2[atts_s2_2_toscale] = scaler_s2_2.transform(X_train_s2_2[atts_s2_2_toscale]).copy()
        X_test_s2_2[atts_s2_2_toscale] = scaler_s2_2.transform(X_test_s2_2[atts_s2_2_toscale]).copy()
        df_train_s2_2 = pd.concat([X_train_s2_2, y_train], axis=1)
        df_test_s2_2 = pd.concat([X_test_s2_2, y_test], axis=1)
        df_train_s2_2.to_csv('{}/{}/{}_{}_RED^OD-BINOD_train.csv'.format(data_fold, filename, filename, meta), index=False)
        df_test_s2_2.to_csv('{}/{}/{}_{}_RED^OD-BINOD_test.csv'.format(data_fold, filename, filename, meta), index=False)
    # RED^RED
    if (os.path.exists(redred_path_test)) & (os.path.exists(redred_path_train)):
        df_train_s2_3 = pd.read_csv(redred_path_train)
        df_test_s2_3 = pd.read_csv(redred_path_test)
        atts_s2_3 = [col for col in df_train_s2_3.columns if (col not in ['TARGET','INDEX']) & (col not in atts_level0)]
        X_train_s2_3 = df_train_s2_3[atts_s2_3].copy()
        X_test_s2_3 = df_test_s2_3[atts_s2_3].copy()
    else:
        s2_3.fit(X_full_s1[atts_s2_red], y_full)
        X_train_s2_3 = s2_3.data_train.iloc[:final_pos].copy()
        X_test_s2_3 = s2_3.data_train.iloc[final_pos:].copy()
        X_test_s2_3.reset_index(drop=True, inplace=True)
        scaler_s2_3 = RobustScaler()
        atts_s2_3_train = [col for col in X_train_s2_3.columns if (col not in atts_s2_red) &
                           (col not in ['INDEX', 'TARGET'])]
        atts_s2_3_test = [col for col in X_test_s2_3.columns if (col not in atts_s2_red) &
                          (col not in ['INDEX', 'TARGET'])]
        atts_s2_3 = sorted(list(set(atts_s2_3_train.copy()).intersection(set(atts_s2_3_test.copy()))), reverse=True)
        atts_s2_3_toscale = [el for el in atts_s2_3 if ('bin_score' not in el)]
        scaler_s2_3.fit(X_train_s2_3[atts_s2_3_toscale])
        X_train_s2_3[atts_s2_3_toscale] = scaler_s2_3.transform(X_train_s2_3[atts_s2_3_toscale]).copy()
        X_test_s2_3[atts_s2_3_toscale] = scaler_s2_3.transform(X_test_s2_3[atts_s2_3_toscale]).copy()
        df_train_s2_3 = pd.concat([X_train_s2_3, y_train], axis=1)
        df_test_s2_3 = pd.concat([X_test_s2_3, y_test], axis=1)
        df_train_s2_3.to_csv('{}/{}/{}_{}_RED^RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
        df_test_s2_3.to_csv('{}/{}/{}_{}_RED^RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
    # OD-BINOD^RED
    if (os.path.exists(odred_path_test)) & (os.path.exists(odred_path_train)):
        df_train_s2_4 = pd.read_csv(odred_path_train)
        df_test_s2_4 = pd.read_csv(odred_path_test)
        atts_s2_4 = [col for col in df_train_s2_4.columns if (col not in ['TARGET']) & (col not in atts_level0)]
        X_train_s2_4 = df_train_s2_4[atts_s2_4].copy()
        X_test_s2_4 = df_test_s2_4[atts_s2_4].copy()
        # time_df = pd.read_csv('{}/{}/{}_{}_RED^OD-BINOD_time.csv'.format(data_fold, filename, filename, meta))
    else:
        s2_4.fit(X_full_s1[atts_s2_od], y_train)
        X_train_s2_4 = s2_4.data_train.iloc[:final_pos].copy()
        X_test_s2_4 = s2_4.data_train.iloc[final_pos:].copy()
        X_test_s2_4.reset_index(drop=True, inplace=True)
        scaler_s2_4 = RobustScaler()
        atts_s2_4_train = [col for col in X_train_s2_4.columns if (col not in atts_s2_od) &
                           (col not in ['INDEX', 'TARGET'])]
        atts_s2_4_test = [col for col in X_test_s2_4.columns if (col not in atts_s2_od) &
                          (col not in ['INDEX', 'TARGET'])]
        atts_s2_4 = sorted(list(set(atts_s2_4_train.copy()).intersection(set(atts_s2_4_test.copy()))), reverse=True)
        atts_s2_4_toscale = [el for el in atts_s2_4 if ('bin_score' not in el)]
        scaler_s2_4.fit(X_train_s2_4[atts_s2_4_toscale])
        try:
            X_train_s2_4[atts_s2_4_toscale] = scaler_s2_4.transform(X_train_s2_4[atts_s2_4_toscale]).copy()
            X_test_s2_4[atts_s2_4_toscale] = scaler_s2_4.transform(X_test_s2_4[atts_s2_4_toscale]).copy()
        except:
            import pdb; pdb.set_trace()
        df_train_s2_4 = pd.concat([X_train_s2_4, y_train], axis=1)
        df_test_s2_4 = pd.concat([X_test_s2_4, y_test], axis=1)
        df_train_s2_4.to_csv('{}/{}/{}_{}_OD-BINOD^RED_train.csv'.format(data_fold, filename, filename, meta), index=False)
        df_test_s2_4.to_csv('{}/{}/{}_{}_OD-BINOD^RED_test.csv'.format(data_fold, filename, filename, meta), index=False)
    # WHOLE DATASET TRAINING
    atts_odod_train = [col for col in X_train_s2_1.columns.tolist() if (col not in atts_level1)].copy()
    atts_redod_train = [col for col in X_train_s2_2.columns.tolist() if (col not in atts_level1)].copy()
    atts_redred_train = [col for col in X_train_s2_3.columns.tolist() if (col not in atts_level1)].copy()
    atts_odred_train = [col for col in X_train_s2_4.columns.tolist() if (col not in atts_level1)].copy()
    if index_train is None:
        db_train = pd.concat([X_train, X_train_s1[atts_level1], X_train_s2_1[atts_odod_train],
                              X_train_s2_2[atts_redod_train], X_train_s2_3[atts_redred_train],
                              X_train_s2_4[atts_odred_train], y_train], axis=1)
    else:
        db_train = pd.concat([index_train, X_train, X_train_s1[atts_level1], X_train_s2_1[atts_odod_train],
                              X_train_s2_2[atts_redod_train], X_train_s2_3[atts_redred_train],
                              X_train_s2_4[atts_odred_train], y_train], axis=1)

    db_train.to_csv('{}/{}/{}_{}_WHOLE_train.csv'.format(data_fold, filename, filename, meta), index=False)
    # WHOLE DATASET TEST
    atts_odod_test = [col for col in X_test_s2_1.columns.tolist() if (col not in atts_level1)].copy()
    atts_redod_test = [col for col in X_test_s2_2.columns.tolist() if (col not in atts_level1)].copy()
    atts_redred_test = [col for col in X_test_s2_3.columns.tolist() if (col not in atts_level1)].copy()
    atts_odred_test = [col for col in X_test_s2_4.columns.tolist() if (col not in atts_level1)].copy()
    if index_test is None:
        db_test = pd.concat([X_test, X_test_s1[atts_level1], X_test_s2_1[atts_odod_test],
                             X_test_s2_2[atts_redod_test], X_test_s2_3[atts_redred_test], X_test_s2_4[atts_odred_test],
                             y_test], axis=1)
    else:
        db_test = pd.concat([index_test, X_test, X_test_s1[atts_level1], X_test_s2_1[atts_odod_test],
                             X_test_s2_2[atts_redod_test], X_test_s2_3[atts_redred_test], X_test_s2_4[atts_odred_test],
                             y_test], axis=1)
    print(db_test.head())
    print(db_train.head())
    db_test.to_csv('{}/{}/{}_{}_WHOLE_test.csv'.format(data_fold, filename, filename, meta), index=False)
    time_s1 = s1.time_to_expand
    time_s2_1 = s2_1.time_to_expand
    time_s2_2 = s2_2.time_to_expand
    time_s2_3 = s2_3.time_to_expand
    time_s2_4 = s2_4.time_to_expand
    time_whole = time_s1 + time_s2_1 + time_s2_2 + time_s2_3 + time_s2_4
    list_times = list(zip(['OD-BINOD-RED', 'OD-BINOD^OD-BINOD', 'RED^OD-BINOD', 'RED^RED', 'OD-BINOD^RED', 'WHOLE'],
                          [time_s1, time_s2_1, time_s2_2, time_s2_3, time_s2_4, time_whole]))
    time_df = pd.DataFrame(list_times, columns=['FROID_method', 'time'])
    time_df.to_csv('{}/{}/{}_{}_time.csv'.format(data_fold, filename, filename, meta), index=False)
    return db_train, db_test


def experiment_FROID(filename, classifier='lgbm', method='WHOLE', feat_sel = 'nofs', full=False,
                     data_fold='data', results_fold='results_xgbod',
                     params='default', calibration='NoCal', meta='FROID', feat_imp=True, methodlist='default'):
    np.random.seed(42)
    random.seed(42)
    print("looking for data in {}".format(data_fold))
    if full:
        meta = 'FROIDFULL'
    if methodlist!="default":
        meta = meta +"_{}".format(methodlist)
        print(meta)
    path_train = '{}/{}/{}_{}_WHOLE_train.csv'.format(data_fold, filename, filename, meta)
    path_test = '{}/{}/{}_{}_WHOLE_test.csv'.format(data_fold, filename, filename, meta)
    if (os.path.exists(path_train)) & (os.path.exists(path_test)):
        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)
    else:
        X_train = pd.read_csv('{}/{}/{}_train.csv'.format(data_fold, filename, filename))
        X_test = pd.read_csv('{}/{}/{}_test.csv'.format(data_fold, filename, filename))
        index_train = X_train[['INDEX']].copy()
        index_test = X_test[['INDEX']].copy()
        y_train = X_train['TARGET'].copy()
        y_test = X_test['TARGET'].copy()
        features = [col for col in X_train.columns.tolist() if (col not in ['TARGET', 'INDEX'])]
        print("experiment FROID methodlist is {}".format(methodlist))
        dict_method_od, dict_method_red = get_dictionaries_methods(ml=methodlist, data_fold=data_fold)
        print(dict_method_od[filename])
        # print(dict_method_red)
        # print(dict_method_od)
        print(filename)
        if full:
            df_train, df_test = get_whole_FROID_FULL(X_train[features], X_test[features], y_train, y_test, dict_method_od,
                                                dict_method_red,
                                                index_train=index_train, index_test=index_test, filename=filename,
                                                data_fold='data', meta=meta)
        else:
            df_train, df_test = get_whole_FROID(X_train[features], X_test[features], y_train, y_test, dict_method_od, dict_method_red,
                        index_train=index_train, index_test=index_test, filename=filename, data_fold = 'data', meta=meta)
    df_train.columns = [col.replace("(","") for col in df_train.columns]
    df_test.columns = [col.replace("(", "") for col in df_test.columns]
    df_train.columns = [col.replace(")","") for col in df_train.columns]
    df_test.columns = [col.replace(")", "") for col in df_test.columns]
    print(df_train.columns)
    orig_cols = [col for col in df_train.columns.tolist() if (col not in ['TARGET', 'INDEX'])
                 & ("scores_" not in col)
                 & ("comp_" not in col)
                 ]
    whole_cols = [col for col in df_train.columns.tolist() if (col not in ['TARGET', 'INDEX']) & ('LDA' not in col)]
    if method == 'WHOLE':
        features = whole_cols.copy()
        if full:
            features = [col for col in features if ('LDA' not in col)]
    elif method == 'ONLY_OD':
        features = [col for col in whole_cols if ('score' in col)]
    elif method == 'ONLY_RED':
        features = [col for col in whole_cols if ('comp' in col)]
    elif method == 'ORIG_OD':
        features = orig_cols.copy() + [col for col in whole_cols if ('score' in col)]
    elif method == 'ORIG_RED':
        features = orig_cols.copy() + [col for col in whole_cols if ('comp' in col)]
    elif method == 'ORIG':
        features = orig_cols.copy()
    elif method == 'NO_ORIG':
        features = [col for col in whole_cols if col not in orig_cols ]
    elif method == 'WHOLE_NOBIN':
        features = [col for col in whole_cols if ('bin_scores' not in col)].copy()
    elif method =='OD-BINOD-RED':
        features = orig_cols+[col for col in whole_cols if ('ORIG' in col)].copy()
    elif method =='OD-BINOD-RED-clean':
        features = orig_cols+[col for col in whole_cols if ('ORIG' in col) & ('LDA' not in col)].copy()
    elif method =='XGBOD':
        features = orig_cols + [col for col in whole_cols if
                                ('ORIG' in col) & ('scores_' in col) & ('bin_' not in col)].copy()
    if classifier == "lgbm":
        clf_base = lgb.LGBMClassifier(importance_type="gain", random_state=42, n_jobs=4)
        if params == 'ht':
            # Define the parameter grid
            param_grid = {
                'learning_rate': [0.01, 0.1, 1],
                'n_estimators': [50, 100, 200],
                'reg_alpha': [0, 1e-1, 1],
                'reg_lambda': [0, 1e-1, 1],
            }
            def auc_score(y_true, y_pred):
                return roc_auc_score(y_true, y_pred)
            grid_search = GridSearchCV(clf_base, param_grid, cv=5, scoring=auc_score)
    elif classifier == 'cat':
        clf_base = CatBoostClassifier(random_state=42, verbose=False, thread_count=4)
    elif classifier == "xgb":
        clf_base = XGBClassifier(importance_type="gain", random_state=42, n_jobs=4)
    elif classifier == "rf":
        clf_base = RandomForestClassifier(random_state=42)
    elif classifier == "dt":
        clf_base = DecisionTreeClassifier(random_state=42)
    features = check_columns(df_train,df_test, features)
    start_time = time.time()
    if params == 'ht':
        print("going to hypterparams tuning")
        grid_search.fit(df_train[features], df_train['TARGET'])
        best_params = grid_search.best_params_
        if classifier == "lgbm":
            clf_base = lgb.LGBMClassifier(importance_type="gain", random_state=42, n_jobs=4, **best_params)
    if calibration == 'calib':
        clf_cal = CalibratedClassifierCV(base_estimator=clf_base, cv=5, method='isotonic')
        feat_imp = False
    if feat_sel == "nofs":
        if calibration == 'calib':
            clf = clf_cal
        else:
            clf = clf_base
    elif feat_sel == "variance":
        selector = VarianceThreshold(threshold=0.2)
        inf_mean_cols = df_train.columns[
            np.where(np.mean(df_train) == np.inf)
        ].tolist()
        inf_var_cols = df_train.columns[
            np.where(np.var(df_train) == np.inf)
        ].tolist()
        features = [
            el
            for el in features
            if (el not in inf_var_cols) and (el not in inf_mean_cols)
        ]
        if calibration == 'calib':
            clf = Pipeline(
                steps=[
                    ("fs", selector),
                    ("clf", clf_cal),
                ]
            )
        else:
            clf = Pipeline(
                steps=[
                    ("fs", selector),
                    ("clf", clf_base),
                ]
            )
    elif (feat_sel == "selectmodel") & (calibration != 'calib'):
        selector = SelectFromModel(clf_base)
        clf = Pipeline(
        steps=[
            ("fs", selector),
            ("clf", clf_base),
            ]
        )
    elif (feat_sel == "selectmodel") & (calibration == 'calib'):
        selector = SelectFromModel(clf_base)
        clf = Pipeline(
            steps=[
                ("fs", selector),
                ("clf", clf_cal),
            ]
        )
        print("I have been here")
        print(calibration)
    if (classifier in ['dt', 'xgb', 'rf']) & ((len(np.where(df_train.max() > 10 ** 20)[0]) > 0)|(
      len(np.where(df_test.max() > 10 ** 20)[0]) > 0)):
        print("values too large for the current classifier: removing below 10**20")
        k1 = df_train.columns[np.where(df_train.max() > 10 ** 20)[0]].tolist()
        k2 = df_test.columns[np.where(df_test.max() > 10 ** 20)[0]].tolist()
        features = [el for el in features if (el not in k2) & (el not in k1)]
    if (classifier in ['dt', 'rf']) & (filename == 'wine_quality'):
        feats_inf = df_train[df_train[features].isna().any()].columns.tolist()
        features = [el for el in features if el not in feats_inf]
    clf.fit(df_train[features], df_train['TARGET'])
    end_time = time.time()
    scores = clf.predict_proba(df_test[features])[:, 1]
    preds = clf.predict(df_test[features])
    y_true = df_test["TARGET"].values
    yy = pd.DataFrame(
        np.c_[y_true, preds, scores], columns=["true", "preds", "scores"]
    )
    results = pd.DataFrame()
    tmp = get_bootstrap_estimates(yy)
    tmp["time_to_fit"] = end_time - start_time
    if isinstance(clf, Pipeline):
        tmp['num_features'] = len(clf['fs'].get_feature_names_out())
    else:
        tmp['num_features'] = len(features)
    tmp["config"] = "{}_{}".format(meta, method)
    tmp["feat_sel"] = feat_sel
    tmp["base_classifier"] = classifier
    tmp["dataset"] = filename
    tmp["parameters"] = params
    tmp["calibration"] = calibration
    results = pd.concat([results, tmp], axis=0)
    results_path = '{}/{}/{}_{}_{}_{}_{}_{}_{}.csv'.format(results_fold,filename, filename,
                                                                   meta, method, classifier, feat_sel, params, calibration)
    if os.path.exists("{}/{}".format(results_fold, filename)):
        results.to_csv(results_path)
    else:
        os.mkdir("{}/{}".format(results_fold, filename))
        results.to_csv(results_path)
    if (feat_imp == True) & (classifier in ['lgbm', 'xgb', 'cat']):
        if isinstance(clf, Pipeline):
            cols_names = clf['fs'].get_feature_names_out(features)
            tmp1 = pd.DataFrame(
                np.c_[cols_names, clf[-1].feature_importances_],
                columns=["features", "feat_importance"],
            )
        elif isinstance(clf, FROID):
            tmp1 = pd.DataFrame(
                np.c_[clf.atts_, clf.clf_base.feature_importances_],
                columns=["features", "feat_importance"],
            )
        elif isinstance(clf, CalibratedClassifierCV):
            print("calibrated classifier")
            cals = [clf.calibrated_classifiers_[i].base_estimator.feature_importances_ for i
                    in range(len(clf.calibrated_classifiers_))]
            col_fi = ['features'] + ["feat_importance_{}".format(i) for i in range(len(clf.calibrated_classifiers_))]
            tt = np.column_stack((clf.calibrated_classifiers_[0].base_estimator.feature_name_, cals
                                  ))

            try:
                tmp1 = pd.DataFrame(
                    tt,
                    columns=col_fi
                )
                print("there we go")
            except:
                tmp1 = pd.DataFrame([['nocol',0]], columns= ["features", "feat_importance"])
        else:
            try:
                if classifier == 'lgbm':
                    cols_names = clf.feature_name_
                else:
                    cols_names = features
                tmp1 = pd.DataFrame(
                    np.c_[cols_names, clf.feature_importances_],
                    columns=["features", "feat_importance"],
                )
            except:
                import pdb;

                pdb.set_trace()

        tmp1["base_classifier"] = classifier
        tmp1["dataset"] = filename
        tmp1["paramters"] = params
        tmp1["calibration"] = calibration
        tmp1.to_csv(
            "{}/{}/FEAT_IMP_{}_{}_{}_{}_{}_{}_{}.csv".format(results_fold,
                filename, filename, meta, classifier, method, feat_sel, params, calibration
            )
    )
    print(methodlist)



def experiment_NOFROID(
    filename,
    classifier,
    method="RED",
    parameters="default",
    binary=True,
    calibration="NoCalibration",
    scaling=False,
    meta="FROID",
    sampling_perc=1,
    feat_imp=False,
    feat_sel="nofs",
    report_time=False,
    data_fold='data'
):
    np.random.seed(42)
    random.seed(42)
    results = pd.DataFrame()
    if classifier == "lgbm":
        if parameters == "default":
            clf_base = lgb.LGBMClassifier(importance_type="gain", random_state=42)
    if meta == 'CURE':
        string_meta = 'CURE_50.0'
    elif meta in ['RBO', 'CCR', 'SWIM']:
        string_meta = meta+"_100"
    else:
        string_meta = meta
    train_path = '{}/{}/{}_{}_train.csv'.format(data_fold, filename, filename, string_meta)
    if os.path.exists(train_path):
        print("training already existed!!!!!!!!!!!!!!!!!")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv('{}/{}/{}_test.csv'.format(data_fold, filename, filename))
        features = [col for col in df_train.columns.tolist() if (col not in ['TARGET', 'INDEX'])]
        clf = clf_base.copy()
        start_time = time.time()
        clf.fit(df_train[features], df_train["TARGET"])
        end_time = time.time()
        if binary:
            scores = clf.predict_proba(df_test[features])[:, 1]
            preds = clf.predict(df_test[features])
            y_true = df_test["TARGET"].values
            metrics = get_results_binary(y_true, scores, preds)
            yy = pd.DataFrame(
                np.c_[y_true, preds, scores], columns=["true", "preds", "scores"]
            )
            tmp = get_bootstrap_estimates(yy)
            tmp["time_to_fit"] = end_time - start_time
            tmp["num_features"] = len(features)
            tmp["config"] = meta
            tmp["feat_sel"] = "nofs"
            tmp["time_to_expand"] = 0
            tmp["base_classifier"] = classifier
            tmp["dataset"] = filename
            tmp["paramters"] = parameters
            tmp["calibration"] = calibration
            results = pd.concat([results, tmp], axis=0)
        if os.path.exists("final_results/{}".format(filename)) == False:
            os.mkdir("final_results/{}".format(filename))
            results.to_csv('final_results/{}/results_{}_{}_{}_{}_{}_{}.csv'.format(filename, filename, classifier, meta,
                                                                                   parameters, calibration, scaling))
        else:
            results.to_csv('final_results/{}/results_{}_{}_{}_{}_{}_{}.csv'.format(filename, filename, classifier, meta,
                                                                                   parameters, calibration, scaling))
        if (feat_imp) & (meta != "FROID"):
            tmp1 = pd.DataFrame(
                np.c_[clf.atts_, clf.clf_base.feature_importances_],
                columns=["features", "feat_importance"],
            )
            tmp1["base_classifier"] = classifier
            tmp1["dataset"] = filename
            tmp1["paramters"] = parameters
            tmp1["calibration"] = calibration
            tmp1.to_csv(
                "results/{}/FEAT_IMP_{}_{}_{}_{}_{}_{}.csv".format(
                    filename, filename, classifier, method, parameters, calibration, scaling
                )
            )
    else:
        if meta == "SMOTE_{}".format(sampling_perc * 100):
            sampler = SMOTE
            clf = CResampler(
                clf_base,
                sampler,
                sampling_strategy=sampling_perc,
                meta=meta,
                write_train=True,
                filename=filename,
            )
        elif meta == "UNDER_{}".format(sampling_perc * 100):
            sampler = RandomUnderSampler
            clf = CResampler(
                clf_base,
                sampler,
                sampling_strategy=sampling_perc,
                meta=meta,
                write_train=True,
                filename=filename,
            )
        elif meta == "OVER_{}".format(sampling_perc * 100):
            sampler = RandomOverSampler
            clf = CResampler(
                clf_base,
                sampler,
                sampling_strategy=sampling_perc,
                meta=meta,
                write_train=True,
                filename=filename,
            )
        elif meta == "ADASYN_{}".format(sampling_perc * 100):
            sampler = ADASYN
            clf = CResampler(
                clf_base,
                sampler,
                sampling_strategy=sampling_perc,
                meta=meta,
                write_train=True,
                filename=filename,
            )
        elif meta == "SVMSMOTE_{}".format(sampling_perc * 100):
            sampler = SVMSMOTE
            clf = CResampler(
                clf_base,
                sampler,
                sampling_strategy=sampling_perc,
                meta=meta,
                write_train=True,
                filename=filename,
            )
        elif meta == "CURE":
            clf = CCure(clf_base, meta=meta, write_train=True, filename=filename)
        elif meta == "SWIM":
            sampler = SwimRBF
            clf = CResampler(
                clf_base, sampler, sampling_strategy=sampling_perc, meta=meta, write_train=True, filename=filename
            )
        elif meta == "CCR":
            sampler = CCR
            clf = CResampler(
                clf_base, sampler,sampling_strategy=sampling_perc, meta=meta, write_train=True, filename=filename
            )
        elif meta == "RBO":
            sampler = RBO
            clf = CResampler(
                clf_base, sampler, sampling_strategy=sampling_perc, meta=meta, write_train=True, filename=filename
            )
        df_train = pd.read_csv('{}/{}/{}_train.csv'.format(data_fold, filename, filename))
        df_test = pd.read_csv('{}/{}/{}_test.csv'.format(data_fold, filename, filename))
        features = [col for col in df_train.columns.tolist() if (col not in ['TARGET', 'INDEX'])]
        start_time = time.time()
        clf.fit(df_train[features], df_train["TARGET"])
        end_time = time.time()
        tmp_time = pd.DataFrame([[filename,end_time-start_time, meta]], columns=[filename,'time','meta'])
        if binary:
            scores = clf.predict_proba(df_test[features])[:, 1]
            preds = clf.predict(df_test[features])
            y_true = df_test["TARGET"].values
            metrics = get_results_binary(y_true, scores, preds)
            yy = pd.DataFrame(
                np.c_[y_true, preds, scores], columns=["true", "preds", "scores"]
            )
            tmp = get_bootstrap_estimates(yy)
            tmp["time_to_fit"] = end_time - start_time
            if meta == "FROID":
                tmp["num_features"] = len(clf.atts_)
                tmp["config"] = "{}_{}".format(meta, method)
                tmp["feat_sel"] = feat_sel

            else:
                tmp["num_features"] = len(features)
                tmp["config"] = meta
                tmp["feat_sel"] = "nofs"

            tmp["time_to_expand"] = clf.time_to_expand
            tmp["base_classifier"] = classifier
            tmp["dataset"] = filename
            tmp["paramters"] = parameters
            tmp["calibration"] = calibration
            results = pd.concat([results, tmp], axis=0)
        if os.path.exists("final_results/{}".format(filename)) == False:
            os.mkdir("final_results/{}".format(filename))
            results.to_csv('final_results/{}/results_{}_{}_{}_{}_{}_{}.csv'.format(filename, filename, classifier, meta,
                                                                             parameters, calibration, scaling))
            tmp_time.to_csv('final_results/{}/{}_{}_time.csv'.format(filename, filename, meta))
        else:
            results.to_csv('final_results/{}/results_{}_{}_{}_{}_{}_{}.csv'.format(filename, filename, classifier, meta,
                                                                                   parameters, calibration, scaling))
            tmp_time.to_csv('final_results/{}/{}_{}_time.csv'.format(filename, filename, meta))
        if (feat_imp) & (meta != "FROID"):
            tmp1 = pd.DataFrame(
                np.c_[clf.atts_, clf.clf_base.feature_importances_],
                columns=["features", "feat_importance"],
            )
            tmp1["base_classifier"] = classifier
            tmp1["dataset"] = filename
            tmp1["paramters"] = parameters
            tmp1["calibration"] = calibration
            tmp1.to_csv(
                "results/{}/FEAT_IMP_{}_{}_{}_{}_{}_{}.csv".format(
                    filename, filename, classifier, method, parameters, calibration, scaling
                )
            )



















