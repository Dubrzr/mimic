import numpy


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


def smooth(x, window_size):
    box = numpy.ones(window_size)/window_size
    return numpy.convolve(x, box, mode='same')


def compute_best_peak(signal, rpeaks, min_gap, max_gap, threshold=None):
    assert signal.shape == rpeaks.shape
    N = len(signal)
    # The neural network returns probabilities that we have a R-peak for each given sample.
    
    # Sometimes it predicts multiple ones '1' side by side,
    # in order to prevent that, the following code computes the best peak
    
    if threshold is not None:
        x = numpy.where(rpeaks>=threshold)
        numpy.put(rpeaks, x, 1.0)
    rpeaks = rpeaks.astype('int32')
        
    # 1- Extract ranges where we have one or many ones side by side (rpeaks locations predicted by NN)
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
    
    #print('Number of peak ranges found:', len(rpeaks_ranges))
    
    smoothed = smooth(signal, 500)
    
    # Compute signal's peaks
    hard_peaks, soft_peaks = find_peaks(signal)
    all_peak_idxs = numpy.concatenate((hard_peaks, soft_peaks)).astype('int64')
    
    # Replace each range of ones by the index of the best value in it
    tmp = set()
    for rp_range in rpeaks_ranges:
        r = numpy.arange(rp_range[0], rp_range[1]+1, dtype='int64')
        vals = signal[r]
        smoothed_vals = smoothed[r]
        p = r[numpy.argmax(numpy.absolute(numpy.asarray(vals)-smoothed_vals))]
        tmp.add(p)

            
    # Replace all peaks by the peak within x-max_gap < x < x+max_gap which have the bigget distance from smooth curve
    dist = numpy.absolute(signal-smoothed) # Peak distance from the smoothed mean
    rpeaks_indexes = set()
    for p in tmp:
        a = max(0, p-max_gap)
        b = min(N, p+max_gap)
        r = numpy.arange(a, b, dtype='int64')
        idx_best = r[numpy.argmax(dist[r])]
        rpeaks_indexes.add(idx_best)
    
    rpeaks_indexes = list(rpeaks_indexes)
    
    # Prevent multiple peaks to appear in the max bpm range (max_gap)
    # If we found more than one peak in this interval, then we choose the peak with the maximum amplitude compared to the mean of the signal
    tmp = numpy.asarray(rpeaks_indexes)
    to_remove = {}
    for idx in rpeaks_indexes:
        if idx in to_remove:
            continue
        r = tmp[numpy.where(numpy.absolute(tmp-idx)<=max_gap)[0]]
        if len(r) == 1:
            continue
        rr = r.astype('int64')
        vals = signal[rr]
        smoo = smoothed[rr]
        the_one = r[numpy.argmax(numpy.absolute(vals-smoo))]
        for i in r:
            if i != the_one:
                to_remove[i] = True
    for v, _ in to_remove.items():
        rpeaks_indexes.remove(v)
    
    return rpeaks_indexes