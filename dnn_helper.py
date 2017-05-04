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
    annsamp = numpy.sort(annsamp)
    result = numpy.zeros(len(tt), dtype='bool')
    j = 0
    tprec = tt[j]
    for i, v in enumerate(annsamp):
        if anntype[i] not in beat_annotations:
            continue
        while True:
            d = False
            if j+1 == len(tt):
                result[j] = 1
                break
            tnow = tt[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    result[j] = 1
                else:
                    result[j+1] = 1
                d = True
            j += 1
            tprec = tnow
            if d:
                break
    return result


def resample(s, annsamp, anntype, fs, fs_target):
    if fs == fs_target:
        y = numpy.zeros(len(s), dtype='bool')
        y[annsamp] = 1
        return s, y
        
    new_length = int(len(s)*fs_target/fs)
    #print('{:,} -> {:,} ({}hz to {}hz)'.format(len(s), new_length, fs, fs_target), end='')
    x, tt = signal.resample(s, num=new_length, t=numpy.arange(len(s)).astype('float64'))
    assert x.shape == tt.shape
    assert numpy.all(numpy.diff(tt) > 0)
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

       
def stepize(x, y, segment_size, segment_step, normalize_steps):
    assert x.shape[0] >= segment_size
    XY = []
    for i in range(0, len(x)+1-segment_size, segment_step):
        if normalize_steps:
            xx = numpy.reshape(normalize(x[i:i+segment_size]), (segment_size, 1))
        else:
            xx = numpy.reshape(x[i:i+segment_size], (segment_size, 1))
        yy = numpy.reshape(y[i:i+segment_size], (segment_size, 1))
        XY.append((xx, yy))
    assert len(XY) == int(x.shape[0]/segment_step)-1
    return XY


def unstepize(y, segment_size, segment_step):
    N = len(y)
    cut = segment_step/2
    
    res = np.empty(shape=(0), dtype=np.float32)
    for i, e in enumerate(y):
        if i == 0:
            res = np.concatenate((res, e[:cut+segment_step]), axis=0)
        elif i == N-1:
            res = np.concatenate((res, e[cut:]), axis=0)
        else:
            res = np.concatenate((res, e[cut:cut+segment_step]), axis=0)
    
    assert res.shape[0] == 2*(segment_step+cut)+(N-3)*segment_step
    return res


def load_example(db, i, fs_target):
    f = data_dir + db + '/' + 'tr_{}_{}hz-normalized.npy'.format(i, fs_target)
    if not os.path.isfile(f):
        transform_example(db, i, fs_target)
    return numpy.load(f, mmap_mode='r', allow_pickle=True)


def load_steps(db, i, fs_target, segment_size, segment_step, normalized_steps, correct_peaks):
    f = data_dir + db + '/' + 'tr_{}_{}hz_{}ssi_{}sst'.format(i, fs_target, segment_size, segment_step)
    if normalized_steps is not None and normalized_steps:
        f += '_normalized'
    if correct_peaks:
        f += '_corrected'
    f += '.npy'
    return numpy.load(f, mmap_mode='r', allow_pickle=True)


def shuffled_examples(fs_target, y_delay, segment_size, segment_step, normalize_steps):
    hhh = []
    ll = 0
    for db, ids in databases:
        for i in ids:
            for j in range(len(load_steps(db, i, fs_target, y_delay, segment_size, segment_step, normalize_steps))):
                hhh.append((db, i, j))
    numpy.random.shuffle(hhh)
    return hhh


def npz_to_npy(f):
    out_f = f[:-4] + '.npy'
    if os.path.isfile(out_f):
        return
    tmp = numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    numpy.save(out_f, tmp)


def transform_example(db, i, fs_target, segment_size=None, segment_step=None, normalize_steps=None, correct_peaks=False):
    assert (segment_size is None and segment_step is None and normalize_steps is None) or \
           (segment_size is not None and segment_step is not None and normalize_steps is not None)
    print(db, i)
    f = data_dir + db + '/' + i
    out_resamp_norm = data_dir + db + '/' + 'tr_{}_{}hz_normalized'.format(i, fs_target)
    out = data_dir + db + '/' + 'tr_{}_{}hz_{}ssi_{}sst'.format(i, fs_target, segment_size, segment_step)
    if normalize_steps is not None and normalize_steps:
        out += '_normalized'
    if correct_peaks:
        out += '_corrected'
        out_resamp_norm += '_corrected'
    out_resamp_norm += '.npy'
    out += '.npy'
    if not os.path.isfile(out_resamp_norm):
        try:
            sig, fields = wfdb.srdsamp(f)
        except Exception as e:
            print(db, i, 'Could not be loaded with wfdb!')
            return
        fs = fields['fs']
        ann = wfdb.rdann(f, 'atr')
        #if not numpy.array_equal(ann.chan, numpy.full(len(ann.chan), ann.chan[0])): # Changing channels though time
        #    print('Example {}/{} not good...'.format(db, i))
        #    return
        print(f, 'resampling...', end='')
        print('annsamp', len(ann.annsamp))
        x, y = resample(s=sig[:, 0], annsamp=ann.annsamp, anntype=ann.anntype, fs=fs, fs_target=fs_target)    
        print('y', numpy.sum(y))
        if numpy.sum(y) > 0:
            min_bpm = 10
            max_bpm = 350
            min_gap = fs_target*60/min_bpm
            max_gap = fs_target*60/max_bpm
            yy = compute_best_peak(x, y, min_gap, max_gap)
            y = numpy.zeros(x.shape[0], dtype='bool')
            y[numpy.asarray(yy, dtype='int32')] = 1
        print('done!')
        print(f, 'normalizing...', end='')
        x = normalize(x)
        print('done!')
        numpy.save(out_resamp_norm, numpy.asarray([x, y]))
    elif not os.path.isfile(out):
        try:
            xy = numpy.load(out_resamp_norm, mmap_mode='r', allow_pickle=True)
        except Exception as e:
            print('failed on {}: {}'.format(out_resamp_norm, e))
            raise e
        x, y = xy
        print(out_resamp_norm)
    if not os.path.isfile(out):
        if len(x) >= segment_size:
            XY = stepize(x, y, segment_size=segment_size,
                         segment_step=segment_step, normalize_steps=normalize_steps)
            numpy.save(out, numpy.asarray(XY))
            print(out)
        else:
            print('Could not stepize', out, 'as its length {} is lower than the minimum required {}.'.format(len(x), segment_size))


def build_dataset(fs_target, segment_size=None, segment_step=None, normalize_steps=None, correct_peaks=False):
    # Do that in //
    # 1. Resample all signals to fs_target
    # 2. Normalize all signals
    # 3. Stepize all signals
    # 4. Write it to disk
    
    inputs = []
    for db, ids in databases:
        for i in ids:
            inputs.append((db, i, fs_target, segment_size, segment_step, normalize_steps, correct_peaks))
    with Pool(3) as p:
        res = p.starmap(transform_example, inputs)


def save_model(out_file, model, params):
    net = {'model': model, 'params': params}
    pickle.dump(net, open(out_file, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    
def load_model(filepath):
    net = pickle.load(open(filepath,'rb'))
    model = net['model']
    params = net['params']
    return model, params

def load_fnpz(f):
    xy = numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    return xy[0,:], xy[1,:]


def load_db(db_name, y_delay, segment_size, segment_step, train_perc):
    XY_file = data_dir + db_name + '/XY_delay0-5000-2500.normalized.dnn.npz'
    if os.path.isfile(XY_file):
        XY = numpy.load(XY_file, mmap_mode='r', allow_pickle=True)['arr_0']
        print('Loaded {} examples of length {} with a delay of {} sample(s).'.format(len(XY), segment_size, y_delay))
    else:
        sigs = []
        for f in glob.glob(data_dir + db_name + '/*.normalized.dnn.npz'):
            x, y = load_fnpz(f)
            sigs.append((x, y))
        XY = []
        for (x, y) in sigs:
            for i in range(0, len(x)+1-segment_size, segment_step):
                if y_delay == 0:
                    XY.append((
                            [[e] for e in x[i:i+segment_size]],
                            [[e] for e in y[i:i+segment_size-y_delay]]))
                else:
                    # We pad y labels with *y_delay* zeros (so that RNNs have some future context to predict classes)
                    XY.append((
                        [[e] for e in x[i:i+segment_size]],
                        numpy.concatenate(([[0] for e in range(y_delay)], [[e] for e in y[i:i+segment_size-y_delay]]))[:segment_size]
                    ))
        random.shuffle(XY)
        numpy.savez_compressed(XY_file, XY)
        print('Constructed {} examples of length {} with a delay of {} sample(s).'.format(len(XY), segment_size, y_delay))

    s = int(len(XY) * train_perc/100)
    trainXY, testXY = XY[:s], XY[s:]
    print('- {} training examples ({}%)'.format(len(trainXY), train_perc))
    print('- {} testing examples ({}%)'.format(len(testXY), 100-train_perc))
    trainXY = numpy.asarray(trainXY)
    testXY = numpy.asarray(testXY)
    return trainXY, testXY



def sa(v, a1, a2):
    return numpy.swapaxes(v, a1, a2)


def find_peaks(x):
    # Definitions:
    # * Hard peak: a peak that is either /\ or \/
    # * Soft peak: a peak that is either /-*\ or \-*/ (In that cas we define the middle of it as the peak)
    tmp = x[1:]
    tmp = numpy.append(tmp, [0])
    tmp = x-tmp
    tmp[numpy.where(tmp>0)] = +1
    tmp[numpy.where(tmp==0)] = 0
    tmp[numpy.where(tmp<0)] = -1
    tmp2 = tmp[1:]
    tmp2 = numpy.append(tmp2, [0])
    tmp = tmp-tmp2
    hard_peaks = numpy.where(numpy.logical_or(tmp==-2,tmp==+2))[0]+1
    soft_peaks = []
    for iv in numpy.where(numpy.logical_or(tmp==-1,tmp==+1))[0]:
        t = tmp[iv]
        i = iv+1
        while True:
            if i==len(tmp) or tmp[i] == -t or tmp[i] == -2 or tmp[i] == 2:
                break
            if tmp[i] == t:
                soft_peaks.append(int(iv+(i-iv)/2))
                break
            i += 1        
    soft_peaks = numpy.asarray(soft_peaks)+1
    return hard_peaks, soft_peaks


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
    for i in range(1, len(rpeaks)-1):
        if tmp and rpeaks[i] > rpeaks[i+1]:
            rpeaks_ranges.append((tmp_idx, i))
            tmp = False
        elif not tmp and rpeaks[i] < 1.0 and rpeaks[i+1] == 1.0:
            tmp = True
            tmp_idx = i+1
    print('lol', len(rpeaks_ranges))
    mean = sum(signal)/len(signal)
    
    # Compute signal's peaks
    hard_peaks, soft_peaks = find_peaks(signal)
    all_peak_idxs = numpy.concatenate((hard_peaks, soft_peaks))
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


def eval_model(test_exs, eval_fun, min_gap, max_gap, left_border, right_border,
               fs_target, y_delay, segment_size, segment_step, normalized_steps,
               plot_examples=True, exs=None, nb=2, threshold=None, nearest_fpr=None, eval_margin=10):
    if exs is None:
        exs = numpy.random.randint(len(test_exs), size=nb).tolist()
    if plot_examples:
        fig, ax = plt.subplots(len(exs), figsize=(30, 10*len(exs)))#, dpi=600)
    
    y_trues = []
    y_preds = []
    
    for i, ex_id in enumerate(exs):
        print('Example {}'.format(ex_id))
        db, k, j = test_exs[ex_id]
        XY = load_steps(db, k, fs_target, y_delay, segment_size, segment_step, normalized_steps)[j]
        X, Y = numpy.reshape(XY[0], (1, 1, 5000)), numpy.reshape(XY[1], (5000,))
        
        print(X.shape, Y.shape)
        
        res = eval_fun(X)
        res = res[0][0]
        
        best_peaks_idxs = compute_best_peak(signal=X[0][0], rpeaks=res, min_gap=min_gap, max_gap=max_gap, threshold=threshold)
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