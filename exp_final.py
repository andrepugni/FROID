from src.classes.utils import *
from tqdm import tqdm
import argparse
import random
import numpy as np

def main(filename, classifier='lgbm', full=False, ml='default', params='default', calibration='NoCal', run='default', data='data'):
    print("main method list is {}".format(ml))
    if run == 'default':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='variance', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='variance', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE_NOBIN', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG', methodlist=ml, full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='variance')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='variance')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE_NOBIN', methodlist=ml, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, methodlist=ml, method='ORIG')
    elif run == 'tab2':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml,
                             feat_sel='variance', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ONLY_OD', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ONLY_RED', methodlist=ml,
                             feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG_OD', methodlist=ml,
                             feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG_RED', methodlist=ml,
                             feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='NO_ORIG', methodlist=ml,
                             feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG', methodlist=ml, full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml,
                             feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ONLY_OD', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ONLY_RED', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG_OD', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG_RED', methodlist=ml,
                             full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='NO_ORIG', methodlist=ml,
                             feat_sel='selectmodel')
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG', methodlist=ml)
    elif run=='short':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='variance', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_sel='variance')
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration, method='WHOLE', methodlist=ml, feat_sel='selectmodel')
    elif run == 'tab1':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='selectmodel')
    elif run == 'single':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='WHOLE', methodlist=ml, feat_imp=False, feat_sel='selectmodel')
    elif run == 'single_or':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs')
    elif run=='nf':
        sampling_perc=1
        metas = ['CURE', 'CCR', 'RBO', 'SWIM',
                 'SMOTE_{}'.format(sampling_perc * 100), 'UNDER_{}'.format(sampling_perc * 100),
                 'OVER_{}'.format(sampling_perc * 100), 'ADASYN_{}'.format(sampling_perc * 100),
                 'SVMSMOTE_{}'.format(sampling_perc * 100)]
        for meta in metas:
            print(meta)
            try:
                experiment_NOFROID(
                    filename,
                    classifier=classifier,
                    meta=meta,
                    sampling_perc=1,
                    feat_imp=False,
                    feat_sel="nofs",
                    report_time=False,
                    data_fold=data
                )
            except:
                continue
    elif run=='rbo':
        sampling_perc=1
        metas = ['RBO']
        for meta in metas:
            print(meta)
            try:
                experiment_NOFROID(
                    filename,
                    classifier=classifier,
                    meta=meta,
                    sampling_perc=1,
                    feat_imp=False,
                    feat_sel="nofs",
                    report_time=False,
                    data_fold=data
                )
            except:
                continue
    elif run == 'xgbod':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='XGBOD', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='XGBOD', methodlist=ml, feat_imp=False, feat_sel='nofs')
    elif run == 'tab3_FR':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG', methodlist=ml,
                             feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='XGBOD', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold='data_light', method='WHOLE', methodlist='light',
                             feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='ORIG', methodlist=ml,
                             feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='OD-BINOD-RED', methodlist=ml, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='XGBOD', methodlist=ml, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, method='WHOLE', methodlist='light',
                             feat_sel='selectmodel', full=True)
    elif run == 'tab1_alt':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED', methodlist=ml, feat_imp=False, feat_sel='selectmodel')
    elif run == 'tab3_alt':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
        else:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs')
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel')

    elif run == 'tab1_alt_clean':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='ORIG', methodlist=ml, feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier='dt', data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier='rf', data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier='xgb', data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier='cat', data_fold=data, params=params, calibration=calibration,
                             method='OD-BINOD-RED-clean', methodlist=ml, feat_imp=False, feat_sel='selectmodel', full=True)

    elif run == 'cal':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold='data_xgbod', params=params, calibration='calib',
                             method='ORIG', methodlist='xgbod', feat_imp=False, feat_sel='nofs', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold='data', params=params, calibration='calib',
                             method='WHOLE', methodlist='light', feat_imp=False, feat_sel='selectmodel', full=True)
            experiment_FROID(filename, classifier=classifier, data_fold='data_xgbod', params=params, calibration='calib',
                             method='WHOLE', methodlist='xgbod', feat_imp=False, feat_sel='selectmodel', full=True)
    elif run == 'fi':
        if full:
            experiment_FROID(filename, classifier=classifier, data_fold=data, params=params, calibration=calibration,
                             method='WHOLE', methodlist='xgbod', feat_imp=True, feat_sel='selectmodel', full=True)





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=str, default='onlytrain')
    parser.add_argument('--starter', type=int, default=0)
    parser.add_argument('--finisher', type=int, default=100)
    parser.add_argument('--ml', type=str, default='default')
    parser.add_argument('--fls', type=str, default='all')
    parser.add_argument('--model', type=str, default='lgbm')
    parser.add_argument('--params', type=str, default='default')
    parser.add_argument('--run', type=str, default='default')
    parser.add_argument('--cal', type=str, default='NoCal')
    parser.add_argument('--data', type=str, default='data')
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    starter = args.starter
    filelist = sorted(os.listdir(args.data), reverse=False)
    finisher = min(len(filelist), args.finisher)
    filelist_ = filelist[starter:finisher]
    ml = args.ml
    print(ml)
    if args.fls == 'onlyres':
        file_res_list = os.listdir('final_results')
        filelist_ = [el for el in filelist_ if el in file_res_list]
    clf_string = args.model
    params = args.params
    run = args.run
    cal = args.cal

    filelist = ['stars', 'oil']
    for f in tqdm(filelist_):
        print(f)
        if args.full == 'full':
            main(f, full=True, ml=ml, classifier=clf_string, params=params, run=run, data=args.data, calibration=cal)
        elif args.full == 'both':
            main(f, ml=ml, classifier=clf_string, params=params, run=run, data=args.data, calibration=cal)
            print(ml)
            main(f, ml=ml, full=True, classifier=clf_string, params=params, run=run, data=args.data, calibration=cal)
        else:
            main(f, ml=ml, classifier=clf_string, params=params, run=run, data=args.data, calibration=cal)