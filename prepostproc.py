import numpy
from scipy import signal

import wfdb
from wfdb import Annotation
from wfdb.processing import normalize, correct_peaks, resample_sig

from file_utils import load_steps
import os

from multiprocessing.pool import Pool


def resample_ann(tt, annsamp):
    tmp = numpy.zeros(len(tt), dtype='int16')
    j = 0
    tprec = tt[j]
    for i, v in enumerate(annsamp):
        while True:
            d = False
            if v < tprec:
                j -= 1
                tprec = tt[j]
                
            if j+1 == len(tt):
                tmp[j] += 1
                break
            
            tnow = tt[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    tmp[j] += 1
                else:
                    tmp[j+1] += 1
                d = True
            j += 1
            tprec = tnow
            if d:
                break
                
    idx = numpy.where(tmp>0)[0].astype('int64')
    res = []
    for i in idx:
        for j in range(tmp[i]):
            res.append(i)
    assert len(res) == len(annsamp)
    return numpy.asarray(res, dtype='int64')
    


def resample_multichan(xs, ann, fs, fs_target, resamp_ann_chan=0):
    # xs: a numpy.ndarray containing the signals as returned by wfdb.srdsamp
    # ann: an Annotation object
    # fs: the current frequency
    # fs_target: the target frequency
    # resample_ann_channel: the signal channel that is used to compute new annotation indexes

    # Resample multiple channels with their annotations

    assert resamp_ann_chan < xs.shape[1]
    assert numpy.array_equal(ann.annsamp, sorted(ann.annsamp))

    lx = []
    lt = None
    for chan in range(xs.shape[1]):
        xx, tt = resample_sig(xs[:, chan], fs, fs_target)
        lx.append(xx)
        if chan == resamp_ann_chan:
            lt = tt
    
    cc = len(ann.annsamp)
    
    new_annsamp = resample_ann(lt, ann.annsamp)
    
    if cc != len(new_annsamp):
        print(cc, numpy.sum(new_annsamp), len(tt))
        
    assert cc == len(new_annsamp)

    new_ann = Annotation(ann.recordname, ann.annotator, new_annsamp, ann.anntype, ann.num, ann.subtype, ann.chan, ann.aux, ann.fs)
    return numpy.column_stack(lx), new_ann


def to_flat(annsamp, size):
    y = numpy.zeros(size, dtype='bool')
    y[annsamp] = 1
    return y


def is_valid_example(x, y):
    # Valid is:
    #  * At least four annotations
    #  * A continuous signal
    if len(numpy.where(y==1)[0]) < 2:
        return False
    if numpy.max(x) == numpy.min(x):
        return False
    if numpy.isnan(x).any():
        return False
    return True


def stepize_x(x, params):
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    cut = int(segment_step / 2)
    assert x.shape[0] >= segment_size
    
    X = [x[0:segment_size]]
    
    nb_blue, remain = divmod(x.shape[0]-cut-segment_step, segment_step)
    
    if remain > 0 and remain < cut:
        nb_blue -= 1
        
    for i in range(nb_blue):
        lb = (i+1)*segment_step
        ub = (i+1)*segment_step + segment_size
        X.append(x[lb:ub])
    
    if remain > 0:
        X.append(x[-segment_size:])
    
    if params['normalize_steps']:
        X = [wfdb.processingnormalize(x) for x in X]
    
    return X


def stepize_xy(x, y, params, check_validity=False):
    assert x.shape == y.shape
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    cut = int(segment_step / 2)
    normalize_steps = params['normalize_steps']
    assert x.shape[0] >= segment_size
    N = x.shape[0]
    XY = [(x[0:segment_size], y[0:segment_size])]
    
    nb_blue, remain = divmod(x.shape[0]-cut-segment_step, segment_step)
    
    if remain > 0 and remain < cut:
        nb_blue -= 1
    
    for i in range(nb_blue):
        lb = (i+1)*segment_step
        ub = (i+1)*segment_step + segment_size
        XY.append((x[lb:ub], y[lb:ub]))
    
    if remain > 0:
        XY.append((x[-segment_size:], y[-segment_size:]))
    
    if check_validity:
        XY = [(x, y) for x, y in XY if is_valid_example(x, y)]
        print('{}/{} valid examples added!'.format(len(XY), int(N/segment_step)-1))
    else:
        print("{} examples added!".format(len(XY)))
        
    if normalize_steps:
        XY = [(normalize(x), y) for x, y in XY]
    
    return XY


def unstepize(y, length, params):
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    cut = int(segment_step/2)
    
    Y = [y[0][:cut+segment_step]]
    
    nb_blue, remain = divmod(length-cut-segment_step, segment_step)
    
    if remain > 0 and remain < cut:
        nb_blue -= 1
        remain += segment_step
    
    for i in range(nb_blue):
        Y.append(y[i+1][cut:cut+segment_step])
    
    if remain > 0:
        Y.append(y[-1][-remain:])
    
    result = numpy.concatenate(Y)
    assert result.shape[0] == length
    
    return result


def shuffled_examples(dbs, params):
    exs = []
    ll = 0
    for db, ids in dbs:
        for i in ids:
            for j in range(len(load_steps(db, i, params))):
                exs.append((db, i, j))
    numpy.random.shuffle(exs)
    return exs


def npz_to_npy(f):
    out_f = f[:-4] + '.npy'
    if os.path.isfile(out_f):
        return
    tmp = numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    numpy.save(out_f, tmp)
    

def transform_example(data_dir, db, i, params):
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    normalize_steps = params['normalize_steps']
    cp = params['correct_peaks']
    fs_target = params['fs_target']
    min_gap = params['min_gap']
    max_gap = params['max_gap']
    beats = params['beats']
    smooth_window_correct = params['smooth_window_correct']
    
    assert (segment_size is None and segment_step is None and normalize_steps is None) or \
           (segment_size is not None and segment_step is not None and normalize_steps is not None)
    f = data_dir + db + '/' + i
    out_resamp_norm = data_dir + db + '/' + 'tr_{}_{}hz_normalized'.format(i, fs_target)
    out = data_dir + db + '/' + 'tr_{}_{}hz_{}ssi_{}sst'.format(i, fs_target, segment_size, segment_step)
    if normalize_steps is not None and normalize_steps:
        out += '_normalized'
    if cp:
        out += '_corrected'
        out_resamp_norm += '_corrected'
    out_resamp_norm += '.npy'
    out += '.npy'
    try:
        if not os.path.isfile(out_resamp_norm):
            print('Starting {}/{} transform...'.format(db, i))
            sig, fields = wfdb.srdsamp(f)
            fs = fields['fs']
            ann = wfdb.rdann(f, 'atr')
            #if not numpy.array_equal(ann.chan, numpy.full(len(ann.chan), ann.chan[0])): # Changing channels though time
            #    print('Example {}/{} not good...'.format(db, i))
            #    return
            old = len(ann.annsamp)
            if fs_target != fs:
                sig, ann = resample_multichan(xs=sig, ann=ann, fs=fs, fs_target=fs_target)
            assert len(ann.annsamp) == old
            y = numpy.zeros(sig.shape[0], dtype='int32')
            if beats is not None:
                beat_ann_indexes = ann.annsamp[numpy.where(numpy.in1d(ann.anntype, beats))[0]]
                y[beat_ann_indexes] = 1
            else:
                y[ann.annsamp] = 1
            assert numpy.sum(y) > 0
            res = []
            for u in range(sig.shape[1]):
                yy = correct_peaks(x=sig[:,u], peaks_indexes=numpy.where(y>0)[0], min_gap=min_gap, max_gap=max_gap, smooth_window=smooth_window_correct)
                yyy = numpy.zeros(sig.shape[0], dtype='bool')
                yyy[numpy.asarray(yy, dtype='int32')] = 1
                assert numpy.sum(yyy) > 1
                res.append(numpy.stack((sig[:, u], yyy)))
            numpy.save(out_resamp_norm, numpy.asarray(res))
        elif not os.path.isfile(out):
            res = numpy.load(out_resamp_norm, mmap_mode='r', allow_pickle=True)
        if not os.path.isfile(out):
            XY = []
            for e in res:
                x, y = e[0], e[1]
                if x.shape[0] >= segment_size:
                    assert numpy.sum(y) > 0
                    XY += stepize_xy(x, y, params, check_validity=True)
                else:
                    print('Could not stepize', out, 'as its length {} is lower than the minimum required {}.'.format(len(x), segment_size))
                    break
            XY = [(numpy.reshape(x, (segment_size,1)), numpy.reshape(y, (segment_size, 1))) for x, y in XY]
            numpy.save(out, numpy.asarray(XY))
    except Exception as e:
        print('failed on {}/{}'.format(db, i))
        raise e
    print(db, i, 'transformed!')


def build_dataset(data_dir, dbs, params, pool_size=None):
    # Do that in //
    # 1. Resample all signals to fs_target
    # 2. Normalize all signals
    # 3. Stepize all signals
    # 4. Write it to disk
    
    inputs = []
    for db, ids in dbs:
        for i in ids:
            inputs.append((data_dir, db, i, params))
    with Pool(pool_size) as p:
        res = p.starmap(transform_example, inputs)