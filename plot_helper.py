import matplotlib.pyplot as plt
import numpy
import wfdb

def peaks_hr(x, peaks_indexes, fs, title, hrs=None, figsize=(20, 10), saveto=None):
    if hrs is None:
        hrs = wfdb.processing.compute_hr(length=x.shape[0], peaks_indexes=peaks_indexes, fs=fs)
    
    N = x.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(x, color='#3979f0', label='Signal')
    ax_left.plot(peaks_indexes, x[peaks_indexes], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(numpy.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()

    