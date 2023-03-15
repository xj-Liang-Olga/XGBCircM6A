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

pos_fa = '/home/li/public/lxj/Dcirc_complement/machine_learning/transcirc_test/pos_1W5.fasta'
neg_fa = '/home/li/public/lxj/Dcirc_complement/machine_learning/transcirc_test/neg_1W5.fasta'
x, y = load_train_val_bicoding(pos_fa, neg_fa)


def calculate_metric(true, pred):
    confusion = confusion_matrix(true, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)

model_parameter = '/home/li/public/lxj/Dcirc_complement/machine_learning/xgb_fold_3.pkl'

clf = XGBClassifier(n_jobs=-1, learning_rate=0.3, tree_method='gpu_hist', alpha=0.1, gamma=0.15,
                        reg_lambda=0.6, max_depth=10, colsample_bytree=0.7, subsample=1,
                        objective="binary:logistic")
model = pickle.load(open(model_parameter, "rb"))

pre_label = model.predict(x)
y_pred_prob = model.predict_proba(x)
y_pred_prob_test = y_pred_prob[:, 1]

print(
        "test_acc = %.4f, test_recall= %.4f, test_sp = %0.4f, test_precision = %.4f, test_f1score = %.4f, test_mcc= %.4f"
        % (accuracy_score(y, pre_label),
           recall_score(y, pre_label), calculate_metric(y, pre_label), precision_score(y, pre_label),
           f1_score(y, pre_label),
           matthews_corrcoef(y, pre_label)))

fpr_test, tpr_test, thresholds_test = roc_curve(y, y_pred_prob_test)
precision_test, recall_test, _ = precision_recall_curve(y, y_pred_prob_test)

tprs = []
ROC_aucs = []
fprArray = []
tprArray = []
thresholdsArray = []
mean_fpr =np.linspace(0, 1, 100)

recall_array = []
precisions = []
PR_aucs = []
precision_array = []
mean_recall = np.linspace(0, 1, 100)

fprArray.append(fpr_test)
tprArray.append(tpr_test)
thresholdsArray.append(thresholds_test)
tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
tprs[-1][0] = 0.0
roc_auc = auc(fpr_test, tpr_test)
ROC_aucs.append(roc_auc)

recall_array.append(recall_test)
precision_array.append(precision_test)
precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
pr_auc = auc(recall_test, precision_test)
PR_aucs.append(pr_auc)



# classes = list(set(y_test))
# fig = plt.figure(0)
# classes = ['nonm6A', 'm6A']
# plt.imshow(confusion, cmap=plt.cm.Blues)
# indices = range(len(confusion))
# plt.xticks(indices, classes)
# plt.yticks(indices, classes)
# plt.colorbar()
# plt.xlabel('pred')
# plt.ylabel('fact')
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[first_index][second_index])
#
# plt.savefig(model_path + '/' + 'hESCs_test_result/test_hunxiao.png')
# plt.close(0)

colors = cycle(['#5f0f40', '#9a031e' ,'#fb8b24', '#e36414', '#0f4c5c', '#4361ee', '#c44536', '#bdb2ff'])
## ROC plot for CV
fig = plt.figure(0)
for i, color in zip(range(len(fprArray)), colors):
    plt.plot(fprArray[i], tprArray[i], lw=1.5, alpha=0.9, color='r',
             label=' (AUC = %0.4f)' % (ROC_aucs[i]))

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.savefig('.' + '/' + 'transcirc_test_result/test_ROC.png')
# plt.close(0)

fig = plt.figure(1)
for i, color in zip(range(len(recall_array)), colors):
    plt.plot(recall_array[i], precision_array[i], lw=1.5, alpha=0.9, color='r',
             label=' (AUPRC = %0.4f)' % ( PR_aucs[i]))
mean_precision = np.mean(precisions, axis=0)
mean_recall = mean_recall[::-1]
PR_mean_auc = auc(mean_recall, mean_precision)
PR_std_auc = np.std(PR_aucs)


plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")

plt.savefig('.' + '/' + 'transcirc_test_result/test_pr.png')
plt.close(0)