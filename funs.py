import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time, datetime

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #    raise ValueError("'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def myplot(sig, fields, figsize=(20,10), title='', smoothing=None, savedir='schemas'):
    if len(sig) > 1000000:
        print("Displaying more than one million values won't be done.")

    plt.clf()
    fig, ax = plt.subplots(sig.shape[1], figsize=figsize)

    t = np.array(range(0,sig.shape[0]))/fields["fs"]

    for ch in range(0, sig.shape[1]):
        ax[ch].plot(t, sig[:,ch])
        if smoothing:
            ax[ch].plot(t, smooth(sig[:,ch], window=smoothing)[5:-5])
        ax[ch].set_xlabel('time in seconds')
        ax[ch].set_ylabel(fields["signame"][ch] + "/" + fields["units"][ch])

    ax[0].set_title(title)
    plt.show(fig)
    if savedir is not None:
        fig.savefig(f'{savedir}/{time.time()}-{title}.png', bbox_inches='tight')
