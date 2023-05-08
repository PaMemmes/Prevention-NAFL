from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

from utils.plots import plot_confusion_matrix



def calc_all(model, x_test, y_test):
    
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='all')
    cohen_kappa = cohen_kappa_score(y_test, preds)
    precision, recall, fscore, support = score(y_test, preds)
    report = classification_report(y_test, preds)
    print(report)
    print('Accuracy', accuracy)
    print('Precision', precision)
    print('Recall', recall)
    print('fscore:', fscore)
    print('Cohen kappa', cohen_kappa)
    plot_confusion_matrix(cm, name='cm')

    return cm, cm_norm, preds
