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



#def sa(v, a1, a2):
#    return numpy.swapaxes(v, a1, a2)


