from sklearn.metrics import roc_curve, auc
from conf import databases, url, data_dir, beat_annotations
import wfdb
import numpy
import json
import os
import glob
import random
import time
import pickle
from scipy import signal
from multiprocessing.pool import Pool

from matplotlib import pyplot as plt
import matplotlib as mpl


def resample_ann(tt, annsamp, anntype):
    # returns a binary array containing zeros everywhere excepted where there is a heart beat (R peak)
    # tt is the second variable returned by signal.resample, and is increasing
    # annsamp is also increasing
    result = numpy.zeros(len(tt), dtype='bool')
    j = 0
    tprec = tt[j]
    for i, v in enumerate(annsamp):
        if anntype[i] not in beat_annotations:
            continue
        while True:
            tnow = tt[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    result[j] = 1
                else:
                    result[j+1] = 1
                break
            j += 1
            tprec = tnow
    return result


def resample(s, annsamp, anntype, fs, fs_target):
    new_length = int(len(s)*fs_target/fs)
    #print('{:,} -> {:,} ({}hz to {}hz)'.format(len(s), new_length, fs, fs_target), end='')
    x, tt = signal.resample(s, num=new_length, t=numpy.arange(len(s)))
    y = resample_ann(tt, annsamp, anntype)
    assert x.shape == y.shape
    return x, y


def normalize(x, lb=0, ub=1):
    mid = ub - (ub - lb) / 2
    min_v = numpy.min(x)
    max_v = numpy.max(x)
    mid_v =  max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return x * coef - (mid_v * coef) + mid

       
def stepize(x, y, y_delay, segment_size, segment_step, normalize_steps):
    XY = []
    for i in range(0, len(x)+1-segment_size, segment_step):
        if normalize_steps:
            xx = numpy.reshape(normalize(x[i:i+segment_size]), (segment_size, 1))
        else:
            xx = numpy.reshape(x[i:i+segment_size], (segment_size, 1))
        if y_delay == 0:
            yy = numpy.reshape(y[i:i+segment_size], (segment_size, 1))
        else:
            # We pad y labels with *y_delay* zeros (so that RNNs have some future context to predict classes)
            yy = numpy.concatenate(([[0] for e in range(y_delay)], [[e] for e in y[i:i+segment_size-y_delay]]))[:segment_size]
        XY.append((xx, yy))
    return XY


def load_example(db, i, fs_target):
    f = data_dir + db + '/' + 'tr_{}_{}hz-normalized.npy'.format(i, fs_target)
    if not os.path.isfile(f):
        transform_example(db, i, fs_target)
    return numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']


def load_steps(db, i, fs_target, y_delay, segment_size, segment_step, normalized_steps):
    f = data_dir + db + '/' + 'tr_{}_{}hz_{}delay_{}ssi_{}sst'.format(i, fs_target, y_delay, segment_size, segment_step)
    if normalized_steps is not None:
        f += '-normalized.npy'
    else:
        f += '.npy'
    return numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    

def npz_to_npy(f):
    out_f = f[:-4] + '.npy'
    if os.path.isfile(out_f):
        return
    tmp = numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    numpy.save(out_f, tmp)
    

def transform_example(db, i, fs_target, y_delay=None, segment_size=None, segment_step=None, normalize_steps=None):
    assert (y_delay is None and segment_size is None and segment_step is None and normalize_steps is None) or \
           (y_delay is not None and segment_size is not None and segment_step is not None and normalize_steps is not None)
    print(db, i)
    f = data_dir + db + '/' + i
    out_resamp_norm = data_dir + db + '/' + 'tr_{}_{}hz-normalized.npy'.format(i, fs_target)
    out = data_dir + db + '/' + 'tr_{}_{}hz_{}delay_{}ssi_{}sst'.format(i, fs_target, y_delay, segment_size, segment_step)
    if normalize_steps is not None:
        out += '-normalized.npy'
    else:
        out += '.npy'
    if not os.path.isfile(out_resamp_norm):
        sig, fields = wfdb.srdsamp(f)
        fs = fields['fs']
        ann = wfdb.rdann(f, 'atr')
        if not numpy.array_equal(ann.chan, numpy.full(len(ann.chan), ann.chan[0])): # Changing channels though time
            print('Example {}/{} not good...'.format(db, i))
            return
        print(f, 'resampling...', end='')
        x, y = resample(s=sig[:, ann.chan[0]], annsamp=ann.annsamp, anntype=ann.anntype, fs=fs, fs_target=fs_target)
        print('done!')
        print(f, 'normalizing...', end='')
        x = normalize(x)
        print('done!')
        #y = numpy.zeros(len(x))
        #print('annsamp', ann.annsamp, 'anntype', ann.anntype, 'subtype', ann.subtype)
        #y[ann.annsamp] = 1
        numpy.savez_compressed(out_resamp_norm, numpy.asarray([x, y]))
    elif y_delay is not None and not os.path.isfile(out):
        try:
            xy = numpy.load(out_resamp_norm, mmap_mode='r', allow_pickle=True)['arr_0']
        except Exception as e:
            print('failed on {}: {}'.format(out_resamp_norm, e))
            raise e
        x, y = xy
        print(out_resamp_norm)
    if y_delay is not None and not os.path.isfile(out):
        if len(x) >= segment_size:
            XY = stepize(x, y, y_delay=y_delay, segment_size=segment_size,
                         segment_step=segment_step, normalize_steps=normalize_steps)
            numpy.save(out, numpy.asarray(XY))
            print(out)
        else:
            print('Could not stepize', out, 'as its length {} is lower than the minimum required {}.'.format(len(x), segment_size))


def build_dataset(fs_target, y_delay=None, segment_size=None, segment_step=None, normalize_steps=None):
    # Do that in //
    # 1. Resample all signals to fs_target
    # 2. Normalize all signals
    # 3. Stepize all signals
    # 4. Write it to disk
    
    inputs = []
    for db, ids in databases:
        for i in ids:
            inputs.append((db, i, fs_target, y_delay, segment_size, segment_step, normalize_steps))
    with Pool() as p:
        res = p.starmap(transform_example, inputs)


def save_model(out_file, model, params):
    net = {'model': model, 'params': params}
    pickle.dump(net, open(out_file, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    
def load_model(filepath):
    net = pickle.load(open(filepath,'rb'))
    model = net['model']
    params = net['params']
    return model, params


def sa(v, a1, a2):
    return numpy.swapaxes(v, a1, a2)


def is_peak(v, i, soft=False):
    if soft:
        return (v[i-1] <= v[i] and v[i] >= v[i+1]) or (v[i-1] >= v[i] and v[i] <= v[i+1])
    else:
        return (v[i-1] < v[i] and v[i] > v[i+1]) or (v[i-1] > v[i] and v[i] < v[i+1])

def compute_best_peak(signal, rpeaks, min_gap, max_gap, threshold=None):
    # The neural network returns probabilities that we have a R-peak for each given sample.
    #  Sometimes it predicts multiple ones '1' side by side,
    #  in order to prevent that, the following code computes the best peak
    assert signal.shape == rpeaks.shape
    if threshold is not None:
        x = numpy.where(rpeaks>=threshold)
        numpy.put(rpeaks, x, 1.0)
    rpeaks = rpeaks.astype('int32')
    x = numpy.where(rpeaks==1.0)
    y = signal[x]
    # Extract ranges where we have many ones side by side (rpeaks locations predicted by NN)
    rpeaks_ranges = []
    tmp = rpeaks[0] == 1
    tmp_idx = 0
    for i in range(1, len(rpeaks)-2):
        if tmp and rpeaks[i] > rpeaks[i+1]:
            rpeaks_ranges.append((tmp_idx, i))
            tmp = False
        elif not tmp and rpeaks[i] < rpeaks[i+1]:
            tmp = True
            tmp_idx = i+1
    mean = sum(signal)/len(signal)
    
    # Compute signal's peaks
    all_peak_idxs = {}
    for i in range(1, len(signal)-1):
        if is_peak(signal, i, soft=True):
            all_peak_idxs[i] = True
    rpeaks_indexes = []
    for rp_range in rpeaks_ranges:
        r = numpy.arange(rp_range[0]-1, rp_range[1]+2)
        vals = signal[r]
        
        f = False
        for i in range(1, len(vals)-1):
            if i in all_peak_idxs:
                rpeaks_indexes.append(r[i])
                f = True
                
        if not f:
            # Take the sample that has the maximum amplitude compared to the mean of the signal
            rpeaks_indexes.append(r[numpy.argmax(numpy.absolute(numpy.asarray(vals)-mean))])
    # If possible, replace non-peaks by the nearest peak in x-max_gap < x < x+max_gap
    for p in rpeaks_indexes:
        if p not in all_peak_idxs:
            tmp = numpy.asarray(list(all_peak_idxs))
            v = tmp[numpy.argmin(numpy.absolute(tmp-p))] # nearest peak index
            if p-max_gap < v < p+max_gap:
                rpeaks_indexes.remove(p)
                rpeaks_indexes.append(v)
    rpeaks_indexes = list(set(rpeaks_indexes))
    
    # Prevent multiple peaks to appear in the max bpm range (max_gap)
    # If we found more than one peak in this interval, then we choose the peak we the maximum amplitude compared to the mean of the signal
    tmp = numpy.asarray(rpeaks_indexes)
    to_remove = {}
    for idx in rpeaks_indexes:
        if idx in to_remove:
            continue
        r = tmp[numpy.where(numpy.absolute(tmp-idx)<=max_gap)[0]]
        if len(r) == 1:
            continue
        vals = signal[r]
        the_one = r[numpy.argmax(numpy.absolute(numpy.asarray(vals)-mean))]
        for i in r:
            if i != the_one:
                to_remove[i] = True
    for v, _ in to_remove.items():
        rpeaks_indexes.remove(v)
                
    return rpeaks_indexes


def roc_auc(y_true, y_pred, margin=0):
    assert y_true.shape == y_pred.shape
    if margin >= 0:
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_true[i] != y_pred[i]:
                min_i = max(0, i-margin)
                max_i = min(len(y_pred), i+margin+1)
                if 1 in y_true[min_i:max_i]:
                    for j in range(min_i, max_i):
                        y_pred[j] = y_true[j]
    rp = len(numpy.where(y_true==1)[0])
    fp = len(numpy.where((y_pred-y_true)==1)[0])
    tp = len(numpy.where((y_true+y_pred)==2)[0])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return rp, fp, tp, fpr, tpr, thresholds, auc(fpr, tpr)


def eval_model(testXY, eval_fun, min_gap, max_gap, left_border, right_border,
               plot_examples=True, exs=None, nb=2, threshold=None, nearest_fpr=None, eval_margin=10):
    if exs is None:
        exs = numpy.random.randint(len(testXY), size=nb).tolist()
    if plot_examples:
        fig, ax = plt.subplots(len(exs), figsize=(30, 10*len(exs)))#, dpi=600)
    
    y_trues = []
    y_preds = []
    
    for i, ex_id in enumerate(exs):
        print('Example {}'.format(ex_id))
        X = sa(testXY[ex_id, 0], 0, 1)
        Y = sa(testXY[ex_id, 1], 0, 1)
        
        res = eval_fun(numpy.asarray([X]))
        res = res[0][0]
        
        best_peaks_idxs = compute_best_peak(signal=X[0], rpeaks=res, min_gap=min_gap, max_gap=max_gap, threshold=threshold)
        best_peaks_vals = X[0][best_peaks_idxs]
        
        y_true = Y[0]
        y_pred = numpy.zeros(len(res))
        y_pred[best_peaks_idxs] = 1
        
        y_trues += y_true.tolist()[left_border:-right_border]
        y_preds += res.tolist()[left_border:-right_border]
        
        rp, fp, tp, fpr, tpr, thresholds, auc = roc_auc(y_true[left_border:-right_border], y_pred[left_border:-right_border], margin=eval_margin)
        
        if plot_examples:
            ax[i].plot(Y[0]+1, color='blue')
            ax[i].plot(X[0], color='green')
            ax[i].plot(res-1, color='red')
            ax[i].plot(best_peaks_idxs, best_peaks_vals, 'r+')
            ax[i].plot([left_border, left_border], [-1, 2], 'm-')
            ax[i].plot([len(res)-right_border, len(res)-right_border], [-1, 2], 'm-')
            ax[i].set_title('Example {} (TP={}/{}, FP={}/0, TPR={}, FPR={})'.format(ex_id, tp, rp, fp, tpr, fpr))
    
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
        
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print('Samples:\t\t{} samples'.format(len(y_trues)))
    print('Beats:')
    print('  - {} labelized'.format(rp))
    print('  - {} detected'.format(fp+tp))
    print('  - TP:  {}/{}'.format(tp, rp))
    print('  - FP:  {}/{}'.format(fp, 0))
    print('  - TPR: {:.4f}'.format(tp/rp))