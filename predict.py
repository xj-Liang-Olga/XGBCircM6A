import time
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, matthews_corrcoef, precision_score, recall_score, f1_score, \
    accuracy_score, precision_recall_curve, average_precision_score
import matplotlib
from matplotlib import pyplot as plt
from itertools import cycle
from xgboost import plot_importance
from collections import Counter
from extract_features import *
from xgboost.sklearn import XGBClassifier
import argparse
import joblib
import pandas as pd
import numpy as np
from numpy import interp
from matplotlib.patches import ConnectionPatch
import pickle


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-predict_fa", "--predict_fasta", action="store", dest='predict_fa', required=True,
                        help="predict fasta file")
    parser.add_argument("-model_path", "--model_path", action="store", dest='model_path', required=True,
                        help="model_path")
    parser.add_argument("-outfile", "--outfile", action="store", dest='outfile', required=True,
                        help="outfile name")

    args = parser.parse_args()


    predict_file = args.predict_fa
    x, fa_header = load_data_bicoding_with_header(predict_file)
    model_path = args.model_path
    model_parameter = model_path + 'xgb_fold_3.pkl'
    clf = XGBClassifier(n_jobs=-1, learning_rate=0.3, tree_method='gpu_hist', alpha=0.1, gamma=0.15,
                        reg_lambda=0.6, max_depth=10, colsample_bytree=0.7, subsample=1,
                        objective="binary:logistic")
    model = pickle.load(open(model_parameter, "rb"))

    pre_label = model.predict(x)
    y_pred_prob = model.predict_proba(x)
    y_pred_prob_test = y_pred_prob[:, 1]

    with open(args.outfile, 'w') as fw:
        fw.write(fa_header + '\t' + str(pre_label) + '\t' + str(y_pred_prob_test) + '\n')
        