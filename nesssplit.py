#!/usr/bin/env python3

from scipy import signal, stats
from scipy.io import wavfile
import argparse
import datetime as dt
import logging as log
import numpy as np
import presets
import sys

'''Complex number conversions'''

def p2r(radii, angles):
    return radii * np.exp(1j*angles)

def r2p(x):
    return abs(x), np.angle(x)

'''Noise stats'''

def hist_laxis(data, n_bins, range_limits):
    '''https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis'''
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    N = data.shape[-1]
    bins = np.linspace(R[0],R[1],n_bins+1)
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts

def rolling_window(a, window):
    '''Copied from https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html'''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def chisquare(xs, sample_size):
    '''
    Calculate a rolling chisquare p-value.
    xs is a time-series of unwrapped phase differences from an STFT frequency bin.
    This returns a (shorter) time-series of p-values,
    each representing the likelihood that the phase difference in the center slot
    xs[sample_size // 2 + 1] is noise.
    '''
    # take a rolling histogram
    hist_range = (-np.pi, np.pi)    
    hist = lambda sample: np.histogram(sample, sample_size, range=hist_range)[0]
    xs_rolling = rolling_window(xs, sample_size)
    xs_rolling_hist = hist_laxis(xs_rolling, sample_size, hist_range)
    # sanity check: all the sums should be the same (same number of values in each histogram)
    #print(np.apply_along_axis(sum, -1, xs_rolling_hist))
    # take a rolling chisquare stat
    return stats.chisquare(xs_rolling_hist, axis=-1).pvalue

def stdout_print(string):
    sys.stdout.write(f'{string:<40}' + '\r')
    sys.stdout.flush()

class NoiseSplitter(object):
    def __init__(self, nffts, bin_ranges, chisquare_sample_sizes, overlap, pvalue_mask_func, window):
        self.nffts = nffts
        self.bin_ranges = bin_ranges
        self.chisquare_sample_sizes = chisquare_sample_sizes
        self.overlap = overlap
        self.pvalue_mask_func = pvalue_mask_func
        self.window = window
        
    def split_noise_band(self, nfft, bin_range, chisquare_sample_size, normalized_input_data, rate, mix_bus):
        noverlap = (self.overlap - 1) * nfft // self.overlap
        stdout_print(f'Taking STFT: nperseg={nfft}, noverlap={noverlap}')
        f, t, Zxx = signal.stft(normalized_input_data, rate, nperseg=nfft, noverlap=noverlap, window=self.window)
        Zxx_mag, Zxx_phase = r2p(Zxx)

        # Extract phase spectrum bins
        low_bin, high_bin = bin_range
        Zxx_phase_bandpass = Zxx_phase[low_bin:high_bin,:]
        
        # Unwrap phases
        stdout_print('Doing phase things')
        Zxx_phase_unwrapped = np.unwrap(Zxx_phase_bandpass)
        Zxx_phase_unwrapped_diffs = np.diff(Zxx_phase_unwrapped, 1)

        # Run the chi-squared test on the unwrapped phase differences
        stdout_print('Running the chi-squared test')
        chisquare_curried = lambda xs: chisquare(xs, chisquare_sample_size)
        p_values = np.apply_along_axis(chisquare_curried, -1, Zxx_phase_unwrapped_diffs)

        # use the p values to mask any sounds within a certain range of noisiness
        stdout_print('Applying masking function')
        mask = self.pvalue_mask_func(p_values)

        # Pad the mask to preserve the original shape
        stdout_print('Padding mask')
        mask_padded = np.zeros(Zxx.shape)
        end_pad = chisquare_sample_size // 2
        start_pad = chisquare_sample_size - end_pad
        mask_padded[low_bin:high_bin,start_pad:-end_pad] = mask

        # apply mask
        stdout_print('Applying mask')
        Zxx_masked = p2r(Zxx_mag*mask_padded, Zxx_phase)
        
        # take the ISTFT
        stdout_print('Taking ISTFT')
        t_masked, x_masked = signal.istft(Zxx_masked, rate, nperseg=nfft, noverlap=noverlap, window=self.window)
        
        if len(mix_bus) < len(x_masked):
            mix_bus_out = np.zeros(len(x_masked))
            mix_bus_out[:len(mix_bus)] += mix_bus
            mix_bus_out += x_masked
        else:
            mix_bus_out = np.copy(mix_bus)
            mix_bus_out[:len(x_masked)] += x_masked
        return mix_bus_out

    def split_noise(self, input_file_path, output_file_path):
        input_sample_rate, raw_input_data = wavfile.read(input_file_path)
        n_input_frames, n_channels = raw_input_data.shape
        input_dtype = raw_input_data.dtype
        log.debug(f'Input channels: {n_channels}')
        log.debug(f'Input frames: {n_input_frames}')
        log.debug(f'Input sample type: {input_dtype}')
        log.debug(f'Input sample rate: {input_sample_rate}')
        max_dtype_val = np.iinfo(input_dtype).max
        normalized_input_data = raw_input_data / max_dtype_val
        output = []
        for channel in range(n_channels):
            log.info(f'Processing channel {channel+1}')
            input_channel = normalized_input_data[:, channel]
            mix_bus = np.zeros(len(raw_input_data), dtype='float64')
            analysis_args = zip(self.nffts, self.bin_ranges, self.chisquare_sample_sizes)
            for nfft, bin_range, chisquare_sample_size in analysis_args:
                log.debug(f'nfft: {nfft}; bin_range: {bin_range}; chisquare sample size: {chisquare_sample_size}; overlap: {self.overlap}')
                mix_bus = self.split_noise_band(nfft, bin_range, chisquare_sample_size, input_channel, input_sample_rate, mix_bus)
            output.append(mix_bus)
        # write audio
        log.debug(f'Writing audio file {output_file_path}')
        audio_array = np.int16(np.array(output).T * max_dtype_val)
        wavfile.write(output_file_path, input_sample_rate, audio_array)

DEFAULT_PRESET = 'sparkle'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infile',
        help='path to 16-bit wave source file')
    parser.add_argument(
        'outfile',
        help='path to write output file')
    parser.add_argument(
        '-p', '--preset',
        default=DEFAULT_PRESET,
        help=f'preset to use, default is {DEFAULT_PRESET}')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help=f'show debugging messages, default is False')
    parser.add_argument(
        '-l', '--log',
        action='store_true',
        help=f'write logging messages to a file, default is False')
    args = parser.parse_args()
    if args.verbose:
        log_level=log.DEBUG
    else:
        log_level=log.INFO
    if args.log:
        now_str = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        log_filename = f'nesssplit_{now_str}.log'
        log.basicConfig(filename=log_filename, level=log_level, format='%(asctime)s %(message)s')
    else:
        log.basicConfig(level=log_level, format='%(asctime)s %(message)s')
    log.debug(f'Loading preset "{args.preset}"')
    preset = presets.preset_dict[args.preset]
    preset.split_noise(args.infile, args.outfile)
