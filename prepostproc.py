import numpy
from scipy import signal

import wfdb
from wfdb import Annotation

from file_utils import load_steps
from rpeaks import compute_best_peak
from sigproc import normalize
import os

from multiprocessing.pool import Pool
    

def resample_ann(tt, annsamp):
    annsamp = numpy.sort(annsamp)
    result = numpy.zeros(len(tt), dtype='bool')
    j = 0
    tprec = tt[j]
    for i, v in enumerate(annsamp):
        while True:
            if j+1 == len(tt):
                result[j] = 1
                break
            d = False
            if v < tt[j+1]:
                result[j] = 1
                d = True
            j += 1
            if d:
                break
    return numpy.where(result==True)[0].astype('int64')


def resample_sig(x, fs, fs_target):
    t = numpy.arange(x.shape[0]).astype('float64')
    
    if fs == fs_target:
        return x, t
    
    new_length = int(x.shape[0]*fs_target/fs)
    xx, tt = signal.resample(x, num=new_length, t=t)
    assert xx.shape == tt.shape and xx.shape[0] == new_length
    assert numpy.all(numpy.diff(tt) > 0)
    return xx, tt


def resample_singlechan(x, ann, fs, fs_target):
    xx, tt = resample_sig(x, fs, fs_target)
    
    new_annsamp = resample_ann(tt, ann.annsamp)
    print(ann.annsamp.shape, new_annsamp.shape)
    assert ann.annsamp.shape == new_annsamp.shape
    
    new_ann = Annotation(ann.recordname, ann.annotator, new_annsamp, ann.anntype, ann.num, ann.subtype, ann.chan, ann.aux, ann.fs)
    return xx, new_ann


def resample_multichan(xs, ann, fs, fs_target, resamp_ann_chan=0):
    # resample_ann_channel is the signal channel that is used to compute new annotation indexes
    assert resamp_ann_chan < xs.shape[1]
    
    lx = []
    lt = None
    for chan in range(xs.shape[1]):
        xx, tt = resample_sig(xs[:, chan], fs, fs_target)
        lx.append(xx)
        if chan == resamp_ann_chan:
            lt = tt
    
    new_annsamp = resample_ann(lt, ann.annsamp)
    assert ann.annsamp.shape == new_annsamp.shape
    
    new_ann = Annotation(ann.recordname, ann.annotator, new_annsamp, ann.anntype, ann.num, ann.subtype, ann.chan, ann.aux, ann.fs)
    return numpy.column_stack(lx), new_ann


def normalize(x, lb=0, ub=1):
    mid = ub - (ub - lb) / 2
    min_v = numpy.min(x)
    max_v = numpy.max(x)
    mid_v =  max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return x * coef - (mid_v * coef) + mid


def to_flat(annsamp, size):
    y = numpy.zeros(size, dtype='bool')
    y[annsamp] = 1
    return y


def is_valid_example(x, y):
    # Valid is:
    #  * At least two annotations
    #  * A continuous signal
    if numpy.sum(y) < 2:
        return False
    if numpy.isnan(x).any():
        return False
    return True


def stepize(x, y, params, check_validity=False):
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    normalize_steps = params['normalize_steps']
    assert x.shape[0] >= segment_size
    print(numpy.sum(y))
    XY = []
    for i in range(0, len(x)+1-segment_size, segment_step):
        if normalize_steps:
            xx = numpy.reshape(normalize(x[i:i+segment_size]), (segment_size, 1))
        else:
            xx = numpy.reshape(x[i:i+segment_size], (segment_size, 1))
        yy = numpy.reshape(y[i:i+segment_size], (segment_size, 1))
        if check_validity:
            if is_valid_example(xx, yy):
                XY.append((xx, yy))
        else:
            XY.append((xx, yy))
            
    if check_validity:
        print('{}/{} valid examples added!'.format(len(XY), int(x.shape[0]/segment_step)-1))
    else:
        assert len(XY) == int(x.shape[0]/segment_step)-1
    
    return XY


def unstepize(y, params):
    segment_size = params['segment_size']
    segment_step = params['segment_step']
    N = len(y)
    cut = segment_step/2
    
    res = numpy.empty(shape=(0), dtype=numpy.float32)
    for i, e in enumerate(y):
        if i == 0:
            res = numpy.concatenate((res, e[:cut+segment_step]), axis=0)
        elif i == N-1:
            res = numpy.concatenate((res, e[cut:]), axis=0)
        else:
            res = numpy.concatenate((res, e[cut:cut+segment_step]), axis=0)
    
    assert res.shape[0] == 2*(segment_step+cut)+(N-2)*segment_step
    return res


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
    correct_peaks = params['correct_peaks']
    fs_target = params['fs_target']
    min_gap = params['min_gap']
    max_gap = params['max_gap']
    beats = params['beats']
    
    assert (segment_size is None and segment_step is None and normalize_steps is None) or \
           (segment_size is not None and segment_step is not None and normalize_steps is not None)
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
    try:
        if not os.path.isfile(out_resamp_norm):
            print('Starting {}/{} transform...'.format(db, i))
            sig, fields = wfdb.srdsamp(f)
            fs = fields['fs']
            ann = wfdb.rdann(f, 'atr')
            #if not numpy.array_equal(ann.chan, numpy.full(len(ann.chan), ann.chan[0])): # Changing channels though time
            #    print('Example {}/{} not good...'.format(db, i))
            #    return
            sig, ann = resample_multichan(xs=sig, ann=ann, fs=fs, fs_target=fs_target)
            y = numpy.zeros(sig.shape[0], dtype='int32')
            if beats is not None:
                beat_ann_indexes = ann.annsamp[numpy.where(numpy.in1d(ann.anntype, beats))[0]]
                y[beat_ann_indexes] = 1
                #print('len(beat_ann_indexes)', len(beat_ann_indexes), 'numpy.sum(y)', numpy.sum(y), 'len(y)', len(y))
            else:
                y[ann.annsamp] = 1
            if numpy.sum(y) > 0:
                yy = compute_best_peak(sig[:,0], y, min_gap, max_gap)
                y = numpy.zeros(sig.shape[0], dtype='bool')
                y[numpy.asarray(yy, dtype='int32')] = 1
            #x = normalize(x)
            numpy.save(out_resamp_norm, numpy.column_stack([sig, y]))
        elif not os.path.isfile(out):
            xy = numpy.load(out_resamp_norm, mmap_mode='r', allow_pickle=True)
            sig, y = xy[:, 0:xy.shape[1]-2], xy[:, xy.shape[1]-1]
        if not os.path.isfile(out):
            for u in range(sig.shape[1]):
                XY = []
                if sig.shape[0] >= segment_size:
                    XY += stepize(sig[:, u], y, params, check_validity=True)
                else:
                    print('Could not stepize', out, 'as its length {} is lower than the minimum required {}.'.format(len(x), segment_size))
                    break
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