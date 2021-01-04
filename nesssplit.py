from scipy import signal, stats
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np

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
        print(f'\tTaking STFT: nperseg={nfft}, noverlap={noverlap}')
        f, t, Zxx = signal.stft(normalized_input_data, rate, nperseg=nfft, noverlap=noverlap, window=self.window)
        Zxx_mag, Zxx_phase = r2p(Zxx)

        # Extract phase spectrum bins
        low_bin, high_bin = bin_range
        Zxx_phase_bandpass = Zxx_phase[low_bin:high_bin,:]
        
        # Unwrap phases
        print('\tDoing phase things')
        Zxx_phase_unwrapped = np.unwrap(Zxx_phase_bandpass)
        Zxx_phase_unwrapped_diffs = np.diff(Zxx_phase_unwrapped, 1)

        # Run the chi-squared test on the unwrapped phase differences 
        print('\tRunning the chi-squared test')
        chisquare_curried = lambda xs: chisquare(xs, chisquare_sample_size)
        p_values = np.apply_along_axis(chisquare_curried, -1, Zxx_phase_unwrapped_diffs)

        # use the p values to mask any sounds within a certain range of noisiness
        print(f'\tApplying masking function')
        mask = self.pvalue_mask_func(p_values)

        # Pad the mask to preserve the original shape
        print('\tPadding mask')
        mask_padded = np.zeros(Zxx.shape)
        end_pad = chisquare_sample_size // 2
        start_pad = chisquare_sample_size - end_pad
        mask_padded[low_bin:high_bin,start_pad:-end_pad] = mask

        # apply mask
        print('\tApplying mask')
        Zxx_masked = p2r(Zxx_mag*mask_padded, Zxx_phase)
        
        # take the ISTFT
        print('\tTaking ISTFT')
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
        # Read an audio file
        print(f'Loading audio file {input_file_path}')
        input_sample_rate, raw_input_data = wavfile.read(input_file_path)
        n_input_frames, n_channels = raw_input_data.shape
        input_dtype = raw_input_data.dtype
        print(f'Input channels: {n_channels}')
        print(f'Input frames: {n_input_frames}')
        print(f'Input sample type: {input_dtype}')
        print(f'Input sample rate: {input_sample_rate}')
        # TODO: handle float input
        max_dtype_val = np.iinfo(input_dtype).max
        normalized_input_data = raw_input_data / max_dtype_val
        output = []
        for channel in range(n_channels):
            input_channel = normalized_input_data[:, channel]
            mix_bus = np.zeros(len(raw_input_data), dtype='float64')
            analysis_args = zip(self.nffts, self.bin_ranges, self.chisquare_sample_sizes)
            for nfft, bin_range, chisquare_sample_size in analysis_args:
                print(f'nfft: {nfft}; bin_range: {bin_range}; chisquare sample size: {chisquare_sample_size}; ')
                mix_bus = self.split_noise_band(nfft, bin_range, chisquare_sample_size, input_channel, input_sample_rate, mix_bus)
            output.append(mix_bus)
        # write audio
        print(f'Writing audio file {output_file_path}')
        audio_array = np.int16(np.array(output).T * max_dtype_val)
        wavfile.write(output_file_path, input_sample_rate, audio_array)


'''Settings'''

# sparkle settings (good for timestretches of pop music)
xs_sparkle, ys_sparkle = zip((0, 1), (1e-7, 1), (1.001e-7, 0), (1, 0))
pvalue_mask_func_sparkle = interp1d(xs_sparkle, ys_sparkle, kind='linear')
nsp_sparkle_preset = NoiseSplitter(
    nffts = [2048],
    bin_ranges = [(0, 1025)],
    chisquare_sample_sizes = [11],
    overlap = 4,
    pvalue_mask_func = pvalue_mask_func_sparkle,
    window = 'hann')

# layered sparkle settings (good for timestretches of noisy sounds)
xs_sparkle2, ys_sparkle2 = zip((0, 1), (1e-7, 1), (1.001e-7, 0), (1, 0))
pvalue_mask_func_sparkle2 = interp1d(xs_sparkle2, ys_sparkle2, kind='linear')
nsp_sparkle2_preset = NoiseSplitter(
    nffts = [8192, 4096, 2048, 1024, 512],
    bin_ranges = [(0, 129), (65, 129), (65, 129), (65, 129), (65, 257)],
    chisquare_sample_sizes = [11, 11, 21, 41, 81],
    overlap = 4,
    pvalue_mask_func = pvalue_mask_func_sparkle2,
    window = 'hann')

# tone settings
xs_tone, ys_tone = zip((0, 1), (1e-9, 1), (1e-8, 0.5), (1e-7, 0.25), (1e-6, 0.125), (1e-5, 0), (1, 0))
pvalue_mask_func_tone = interp1d(xs_tone, ys_tone, kind='linear')
nsp_tone_preset = NoiseSplitter(
    nffts = [8192, 4096, 2048],
    # cut out the top two octaves
    bin_ranges = [(0, 129), (65, 129), (65, 257)],
    chisquare_sample_sizes = [21, 41, 81],
    overlap = 4,
    pvalue_mask_func = pvalue_mask_func_tone,
    window = 'hann')

# noise settings
xs_noise, ys_noise = zip((0, 0), (0.001, 0), (0.01, 1), (1, 1))
pvalue_mask_func_noise = interp1d(xs_noise, ys_noise, kind='linear')
nsp_noise_preset = NoiseSplitter(
    nffts = [8192, 4096, 2048, 1024, 512, 256],
    bin_ranges = [(0, 129), (65, 129), (65, 129), (65, 129), (65, 129), (65, 129)],
    chisquare_sample_sizes = [11, 11, 11, 11, 11, 11],
    overlap = 4,
    pvalue_mask_func = pvalue_mask_func_noise,
    window = 'hann')

filename = 'audio/bells/197861__bigben12345__kolner-dom-plenum'
input_file_path = f'{filename}.wav'
output_file_path_tone = f'{filename}_tone.wav'
output_file_path_noise = f'{filename}_noise.wav'
#nsp_tone_preset.split_noise(input_file_path, output_file_path_tone)
#nsp_noise_preset.split_noise(input_file_path, output_file_path_noise)
nsp_sparkle2_preset.split_noise(input_file_path, output_file_path_tone)

# TODO: add a multifile export setting
# TODO: write docs
