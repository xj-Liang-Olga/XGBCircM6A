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
# import joblib
import pandas as pd
import numpy as np
from numpy import interp
from matplotlib.patches import ConnectionPatch
import pickle

def calculate_metric(gt, pred):
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)

pos_train_fa = 'train_data/cd80%_RMVar+-.fasta'
neg_train_fa = 'train_data/neg_RMvarnom6A_43819.fasta'

x, y = load_train_val_bicoding(pos_train_fa, neg_train_fa)

neigh = XGBClassifier(n_jobs=-1, learning_rate=0.3, tree_method='gpu_hist', alpha=0.1, gamma=0.15,
                        reg_lambda=0.6, max_depth=10, colsample_bytree=0.7, subsample=1,
                        objective="binary:logistic")

cv = StratifiedKFold(n_splits=5)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['#C3E298', '#A2BDF6', '#E0B9CA', '#EDBD69', '#D57A78'])  # ,'darkorange'
lw = 2

i = 0
plt.figure(0)

fig, ax = plt.subplots(1, 1)
axins = ax.inset_axes((0.6, 0.4, 0.3, 0.3))

# for i, color in zip(range(len(fprArray)), colors):
#         axins.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
#                    label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))

for (train, test), color in zip(cv.split(x, y), colors):
    probas_ = neigh.fit(x[train], y[train]).predict_proba(x[test])
    probas = np.argmax(probas_, axis=1)
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (AUC = %0.4f)' % (i + 1, roc_auc))
    axins.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (AUC = %0.4f)' % (i + 1, roc_auc))

    # neigh.save_model('xgbloost_classifier_model_fold_%d.model' % (i + 1))
    pickle.dump(neigh, open("xgb_fold_%d.pkl" % (i + 1), "wb"))

    i += 1
    print(
        "test_acc = %.4f, test_recall= %.4f, test_sp = %0.4f, test_precision = %.4f, test_f1score = %.4f, test_mcc= %.4f"
        % (accuracy_score(y[test], probas),
           recall_score(y[test], probas), calculate_metric(y[test], probas), precision_score(y[test], probas),
           f1_score(y[test], probas),
           matthews_corrcoef(y[test], probas)))
# plt.plot([0,1],[0,1],linestyle='--',lw=lw,color='k',label='Luck')

mean_tpr /= cv.get_n_splits(x, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr,mean_tpr,color='g',linestyle='--',label='Mean ROC (AUC = %0.2f)'%mean_auc, lw=lw)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
# 设置放大区间
# X轴的显示范围
xlim0 = 0.02
xlim1 = 0.04
# Y轴的显示范围
ylim0 = 0.97
ylim1 = 0.99
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim1,ylim1)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim0,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)
# plt.show()
plt.savefig('./5_folds_CV_result/XGBoost_5fols_roc1.png')
plt.close(0)

mean_precision = 0.0
mean_recall = np.linspace(0, 1, 100)
mean_average_precision = []
plt.figure(1)

i = 0
fig, ax = plt.subplots(1, 1)
axins = ax.inset_axes((0.6, 0.3, 0.3, 0.3))
for (train, test), color in zip(cv.split(x, y), colors):
    y_scores = neigh.fit(x[train], y[train]).predict_proba(x[test])
    # y_scores = max_ent.fit(X_train, y_train).decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y[test], y_scores[:, 1])
    average_precision = average_precision_score(y[test], y_scores[:, 1])
    mean_average_precision.append(average_precision)
    mean_precision += interp(mean_recall, recall, precision)
    pr_auc = auc(recall, precision)
    plt.plot(precision, recall, lw=lw, color=color, label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, average_precision))
    # print(i+1,average_precision)
    # print(i+1,pr_auc)
    axins.plot(precision, recall, lw=lw, color=color, label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, average_precision))
    i += 1

# plt.plot([0,1],[0,1],linestyle='--',lw=lw,color='k',label='Luck')

mean_precision /= cv.get_n_splits(x, y)

mean_average_precision = sum(mean_average_precision) / len(mean_average_precision)

# plt.plot(mean_recall, mean_precision)
# plt.title('PR')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
# plt.show()
# X轴的显示范围
xlim0 = 0.96
xlim1 = 0.98

# Y轴的显示范围

ylim0 = 0.96
ylim1 = 0.98

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

plt.savefig('./5_folds_CV_result/XGBoost_5fols_pr1.png')
plt.close(0)