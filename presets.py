from scipy.interpolate import interp1d
from nesssplit import NoiseSplitter

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

preset_dict = {
    'sparkle': nsp_sparkle_preset,
    'sparkle2': nsp_sparkle2_preset,
    'tone': nsp_sparkle2_preset,
    'noise': nsp_noise_preset
}
