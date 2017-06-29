import numpy
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import matplotlib as mpl
from wfdb.processing import correct_peaks

from file_utils import load_steps


def fix_margin_eval(true, pred, margin):
    for i in range(len(pred)):
        if pred[i] == 1 and true[i] != pred[i]:
            min_i = max(0, i-margin)
            max_i = min(len(pred), i+margin+1)
            if 1 in true[min_i:max_i]:
                for j in range(min_i, max_i):
                    pred[j] = true[j]
    return true, pred

def roc_auc(y_true, y_pred, margin=0):
    assert y_true.shape == y_pred.shape
    if margin >= 0:
        y_true, y_pred = fix_margin_eval(y_true, y_pred, margin)
    rp = len(numpy.where(y_true==1)[0])
    fp = len(numpy.where((y_pred-y_true)==1)[0])
    tp = len(numpy.where((y_true+y_pred)==2)[0])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return rp, fp, tp, fpr, tpr, thresholds, auc(fpr, tpr)

def plot_roc(fpr, tpr, auc, figsize=None):
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def eval_model(test_exs, eval_fun, params, plot_examples=True, exs=None, nb=2, threshold=None, nearest_fpr=None, eval_margin=10):
    assert threshold is not None or nearest_fpr is not None
    min_gap = params['min_gap']
    max_gap = params['max_gap']
    left_border = params['left_border']
    right_border = params['right_border']
    fs_target = params['fs_target']
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    normalize_steps = params['normalize_steps']
    smooth_window_correct = params['smooth_window_correct']
    
    if exs is None:
        exs = numpy.random.randint(len(test_exs), size=nb).tolist()
    if plot_examples:
        fig, ax = plt.subplots(len(exs), figsize=(30, 10*len(exs)))#, dpi=600)
    
    y_trues = []
    y_preds = []
    
    print('Evaluating', end='')
    for i, ex_id in enumerate(exs):
        print('.', end='')
        #print('Example {}'.format(ex_id))
        db, k, j = test_exs[ex_id]
        XY = load_steps(db, k, params)[j]
        X, Y = numpy.reshape(XY[0], (1, 1, 5000)), numpy.reshape(XY[1], (5000,))
        
        #print(X.shape, Y.shape)
        
        res = eval_fun(X)
        res = res[0][0]
        
        if threshold is not None:
            x = numpy.where(res>=threshold)
            numpy.put(res, x, 1.0)
        res = res.astype('int32')
        
        best_peaks_idxs = correct_peaks(x=X[0][0], peak_indexes=res, 
                                                        min_gap=min_gap, max_gap=max_gap, smooth_window=smooth_window_correct)
        best_peaks_vals = X[0][0][best_peaks_idxs]
        
        y_true = Y
        y_pred = numpy.zeros(len(res))
        y_pred[best_peaks_idxs] = 1
        
        y_trues += y_true.tolist()[left_border:-right_border]
        y_preds += res.tolist()[left_border:-right_border]
        
        rp, fp, tp, fpr, tpr, thresholds, auc = roc_auc(y_true[left_border:-right_border], y_pred[left_border:-right_border], margin=eval_margin)
        
        if plot_examples:
            ax[i].plot(Y+1, color='blue')
            ax[i].plot(X[0][0], color='green')
            ax[i].plot(res-1, color='red')
            b_peaks_idx = numpy.where(Y==1)[0]
            ax[i].plot(b_peaks_idx, X[0][0][b_peaks_idx], 'b+')
            ax[i].plot(best_peaks_idxs, best_peaks_vals, 'r+')
            ax[i].plot([left_border, left_border], [-1, 2], 'm-')
            ax[i].plot([len(res)-right_border, len(res)-right_border], [-1, 2], 'm-')
            ax[i].set_title('Example {} ({}/{}/{}) (TP={}/{}, FP={}/0, TPR={}, FPR={})'.format(ex_id, db, k, j, tp, rp, fp, tpr, fpr))
    print()
    if plot_examples:
        plt.show()
    
    rp, fp, tp, fpr, tpr, thresholds, auc = roc_auc(numpy.asarray(y_trues), numpy.asarray(y_preds), margin=eval_margin)
    
    print('FPR\t\t\tTPR\t\t\tThreshold')
    if nearest_fpr is not None:
        idx = (numpy.abs(fpr-nearest_fpr)).argmin()
        print('{:.6f}\t\t{:.6f}\+t\t{:.7f}'.format(fpr[idx], tpr[idx], thresholds[idx]))
    else:
        for i, t in enumerate(thresholds):
            print('{:.6f}\t\t{:.6f}\+t\t{:.7f}'.format(fpr[i], tpr[i], t))
    
    plot_roc(fpr, tpr, auc, figsize=(10,10))
    
    print('Samples:\t\t{} samples'.format(len(y_trues)))
    print('Beats:')
    print('  - {} labelized'.format(rp))
    print('  - {} detected'.format(fp+tp))
    print('  - TP:  {}/{}'.format(tp, rp))
    print('  - FP:  {}/{}'.format(fp, 0))
    print('  - TPR: {:.4f}'.format(tp/rp))