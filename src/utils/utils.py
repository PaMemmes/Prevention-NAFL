from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve

from utils.plots import plot_confusion_matrix

def calc_metrics(confusion_matrix):
    met = defaultdict()
    
    FP = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    FN = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix))
    TP = np.diag(confusion_matrix)
    TN = (confusion_matrix.sum() - (FP + FN + TP))

    FP = FP[1]
    FN = FN[1]
    TP = TP[1]
    TN = TN[1]

    met['TP'] = int(TP)
    met['TN'] = int(TN)
    met['FP'] = int(FP)
    met['FN'] = int(FN)

    met['TPR'] = (TP/(TP+FN)).tolist()
    met['TNR'] = (TN/(TN+FP)).tolist()
    met['PPV'] = (TP/(TP+FP)).tolist()
    met['NPV'] = (TN/(TN+FN)).tolist()
    met['FPR'] = (FP/(FP+TN)).tolist()
    met['FNR'] = (FN/(TP+FN)).tolist()
    met['FDR'] = (FP/(TP+FP)).tolist()

    met['BACC'] = (met['TPR'] + met['TNR']) / 2
    met['ACC'] = ((TP+TN)/(TP+FP+FN+TN)).tolist()

    return met

def calc_all(model, x_test, y_test):
    
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='all')
    accuracy = accuracy_score(y_test, preds)

    metrics = calc_metrics(cm)
    d = dict(metrics)  
    plot_confusion_matrix(cm, name='cm')

    return d, cm, cm_norm, preds